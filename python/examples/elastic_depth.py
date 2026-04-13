"""Elastic depth: train K=4, then attach (K=8) or detach (K=2) from checkpoint."""
import math, sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ski_creche import (
    Block, DataStream, SnowballConfig, Snowman, SnowballTrainer,
    TrainLogger, flop_ratio
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 256
BATCH = 32
D = 384
D_FF = 1536
N_HEADS = 6
LR = 3e-4
WD = 0.01

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elastic.db')
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elastic_ckpts')


class FineWebStream(DataStream):
    def __init__(self, batch, seq_len, split='train', seed=0):
        from datasets import load_dataset
        self.ds = load_dataset(
            'HuggingFaceFW/fineweb-edu', name='sample-10BT',
            split=split, streaming=True, trust_remote_code=True
        ).shuffle(seed=seed, buffer_size=10_000)
        self.enc = tiktoken.get_encoding('gpt2')
        self.batch = batch
        self.seq_len = seq_len
        self.buffer = []
        self.iter = iter(self.ds)

    def _fill_buffer(self, need):
        while len(self.buffer) < need:
            try:
                doc = next(self.iter)
                self.buffer.extend(self.enc.encode_ordinary(doc['text']))
            except StopIteration:
                self.iter = iter(self.ds)

    def get_batch(self):
        need = self.batch * (self.seq_len + 1)
        self._fill_buffer(need)
        tokens = self.buffer[:need]
        self.buffer = self.buffer[need:]
        t = torch.tensor(tokens, dtype=torch.long).view(self.batch, self.seq_len + 1)
        return t[:, :-1].to(DEVICE), t[:, 1:].to(DEVICE)


class TransformerBlock(Block):
    def __init__(self, d, d_ff, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)

    def forward(self, h):
        T = h.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h = h + self.attn(self.ln1(h), self.ln1(h), self.ln1(h),
                          attn_mask=mask, is_causal=True)[0]
        h = h + self.ff2(F.gelu(self.ff1(self.ln2(h))))
        return h


class Embed(nn.Module):
    def __init__(self, vocab_size, d, seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        pe = torch.zeros(seq_len, d)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, idx):
        B, T = idx.shape
        return self.tok(idx) + self.pe[:, :T]


class Readout(nn.Module):
    def __init__(self, d, embed_weight):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        object.__setattr__(self, 'tied_weight', embed_weight)
    def forward(self, h):
        return F.linear(self.ln(h), self.tied_weight)


def build_model(k_max):
    torch.manual_seed(42)
    embed = Embed(50257, D, SEQ_LEN)
    block = TransformerBlock(D, D_FF, N_HEADS)
    readout = Readout(D, embed.tok.weight)
    return Snowman(embed, block, readout, k_max=k_max)


def make_callbacks(model, label):
    def on_step(step, K, loss, lr, flops):
        if step % 200 == 0:
            gv = ','.join(f'{g:.3f}' for g in model.gate_values[:K])
            print(f"  [{label}] step {step:>5} | CE={loss:.4f} | K={K} | lr={lr:.2e} | α=[{gv}]",
                  flush=True)
    def on_eval(step, K, val_ce, flops):
        print(f"  [{label}] eval {step}: K{K}={val_ce:.4f}", flush=True)
    def on_phase(old_K, new_K, step, flops):
        print(f"  [{label}] GROW: K={old_K}->{new_K} at step {step}", flush=True)
    return on_step, on_eval, on_phase


def phase1_train(train_data, eval_data):
    """Train K=1->4, save checkpoint."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Snowball K=1->4")
    print(f"{'='*60}")

    model = build_model(k_max=4)
    on_step, on_eval, on_phase = make_callbacks(model, 'base')

    cfg = SnowballConfig(
        k_max=4, lr=LR, weight_decay=WD, grad_clip=1.0,
        total_steps=500, eval_interval=200, log_interval=200,
        eval_batches=10, seed=42,
        db_path=DB_PATH, run_name='base_K4',
        checkpoint_dir=CKPT_DIR,
    )

    trainer = SnowballTrainer(
        model, cfg, train_data, eval_data,
        tokens_per_step=BATCH * SEQ_LEN,
        on_step=on_step, on_eval=on_eval, on_phase=on_phase,
    )
    result = trainer.train()
    print(f"  base K=4: best={result['best_loss']:.4f} final={result['final_loss']:.4f}")

    # save final checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, 'base_K4_final.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"  saved: {ckpt_path}")
    return ckpt_path, result


def phase2_attach(ckpt_path, train_data, eval_data):
    """Load K=4 checkpoint, resize to K=8, continue growing."""
    print(f"\n{'='*60}")
    print(f"  Phase 2a: Attach K=4->8 (grow)")
    print(f"{'='*60}")

    model = build_model(k_max=4)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.resize_depth(8)
    print(f"  gates after resize: {[f'{g:.3f}' for g in model.gate_values]}")

    on_step, on_eval, on_phase = make_callbacks(model, 'attach')

    cfg = SnowballConfig(
        k_max=8, lr=LR, weight_decay=WD, grad_clip=1.0,
        total_steps=500, eval_interval=200, log_interval=200,
        eval_batches=10, seed=42,
        db_path=DB_PATH, run_name='attach_K8',
        start_k=5,
    )

    trainer = SnowballTrainer(
        model, cfg, train_data, eval_data,
        tokens_per_step=BATCH * SEQ_LEN,
        on_step=on_step, on_eval=on_eval, on_phase=on_phase,
    )
    result = trainer.train()
    print(f"  attach K=8: best={result['best_loss']:.4f} final={result['final_loss']:.4f}")
    return result


def phase2_detach(ckpt_path, train_data, eval_data):
    """Load K=4 checkpoint, infer at K=2 (detach upper depths)."""
    print(f"\n{'='*60}")
    print(f"  Phase 2b: Detach K=4->2 (shrink)")
    print(f"{'='*60}")

    model = build_model(k_max=4)
    state = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # evaluate at K=1,2,3,4 — no retraining, just truncate
    loss_fn = lambda logits, targets: F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1))

    print(f"  gates: {[f'{g:.3f}' for g in model.gate_values]}")
    for k in range(1, 5):
        total = 0.0
        with torch.no_grad():
            for _ in range(20):
                x, y = eval_data.get_batch()
                logits = model(x, k)
                total += loss_fn(logits, y).item()
        ce = total / 20
        print(f"  eval K={k}: CE={ce:.4f}")


def main():
    train_data = FineWebStream(BATCH, SEQ_LEN, split='train', seed=0)
    eval_data = FineWebStream(BATCH, SEQ_LEN, split='train', seed=99999)

    # 1. base training K=1->4
    ckpt_path, base_result = phase1_train(train_data, eval_data)

    # 2a. attach: load K=4, grow to K=8
    attach_result = phase2_attach(ckpt_path, train_data, eval_data)

    # 2b. detach: load K=4, evaluate at K=1,2,3,4 without retraining
    phase2_detach(ckpt_path, train_data, eval_data)

    # summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  base   K=4: best={base_result['best_loss']:.4f}")
    print(f"  attach K=8: best={attach_result['best_loss']:.4f}")
    print(f"  detach: see per-K eval above (no retraining needed)")


if __name__ == '__main__':
    main()
