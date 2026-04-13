"""Ski Creche on FineWeb-Edu: Snowball vs E2E — d=768 scale."""
import math
import sys, os, sqlite3, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ski_creche import (
    Block, DataStream, SnowballConfig, Snowman, SnowballTrainer,
    TrainLogger, flop_ratio, flop_e2e
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 256
BATCH = 16
D = 768
D_FF = 3072
K_MAX = 8
N_HEADS = 12
STEPS_E2E = 2000
LR = 3e-4
WD = 0.01
EVAL_INTERVAL = 200
EVAL_BATCHES = 10

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fineweb_768.db')


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


class E2EModel(nn.Module):
    def __init__(self, vocab_size, d, d_ff, n_heads, n_layers, seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        pe = torch.zeros(seq_len, d)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.blocks = nn.ModuleList([TransformerBlock(d, d_ff, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head_weight = self.tok.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pe[:, :T]
        for blk in self.blocks:
            x = blk(x)
        return F.linear(self.ln_f(x), self.head_weight)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def loss_fn(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def train_e2e(logger, train_data, eval_data, V):
    run = 'e2e_untied'
    print(f"\n{'='*60}")
    print(f"  E2E Untied Transformer: {K_MAX} layers, d={D}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = E2EModel(V, D, D_FF, N_HEADS, K_MAX, SEQ_LEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    print(f"  params={model.param_count:,}", flush=True)

    P_emb = model.tok.weight.numel()
    B_block = sum(p.numel() for p in model.blocks[0].parameters())
    P_head = sum(p.numel() for p in [model.ln_f.weight, model.ln_f.bias]) + V * D
    fpt = flop_e2e(K_MAX, P_emb, B_block, P_head)
    tps = BATCH * SEQ_LEN

    t0 = time.time()
    flops_cum = 0.0
    best_ce = float('inf')

    for step in range(STEPS_E2E):
        lr = LR * 0.5 * (1 + np.cos(np.pi * step / STEPS_E2E))
        for pg in opt.param_groups:
            pg['lr'] = lr

        x, y = train_data.get_batch()
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        flops_cum += fpt * tps

        if step % 200 == 0:
            logger.log_step(run, step, K_MAX, loss.item(), lr, flops_cum, [], [])
            logger.commit()
            print(f"  step {step:>5} | CE={loss.item():.4f} | lr={lr:.2e}", flush=True)

        if (step + 1) % EVAL_INTERVAL == 0:
            model.eval()
            val = 0.0
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    x, y = eval_data.get_batch()
                    val += loss_fn(model(x), y).item()
            val /= EVAL_BATCHES
            model.train()
            if val < best_ce:
                best_ce = val
            logger.log_eval(run, step + 1, K_MAX, val, flops_cum)
            logger.commit()
            print(f"  >>> eval {step+1}: CE={val:.4f} (best={best_ce:.4f})", flush=True)

    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    logger.log_summary(run, 'best_ce', f'{best_ce:.6f}')
    logger.log_summary(run, 'params', str(model.param_count))
    logger.log_summary(run, 'total_flops', f'{flops_cum:.3e}')
    logger.log_summary(run, 'elapsed', f'{elapsed:.1f}')
    logger.log_summary(run, 'peak_vram_mb', f'{peak_mb:.1f}')
    logger.commit()
    print(f"  Final: best={best_ce:.4f}, time={elapsed:.0f}s, FLOP={flops_cum:.2e}, VRAM={peak_mb:.0f}MB", flush=True)

    ret = flops_cum
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return ret


def main():
    V = 50257
    logger = TrainLogger(DB_PATH)

    print(f"d={D}, d_ff={D_FF}, heads={N_HEADS}, K_max={K_MAX}, seq={SEQ_LEN}, batch={BATCH}")

    train_data = FineWebStream(BATCH, SEQ_LEN, split='train', seed=0)
    eval_data = FineWebStream(BATCH, SEQ_LEN, split='train', seed=99999)

    # E2E baseline — skip if already in DB
    row = logger.conn.execute(
        "SELECT value FROM summary WHERE run='e2e_untied' AND key='total_flops'"
    ).fetchone()
    if row:
        flop_budget = float(row[0])
        print(f"  E2E already in DB, FLOP budget={flop_budget:.3e}")
    else:
        flop_budget = train_e2e(logger, train_data, eval_data, V)

    # Snowball
    print(f"\n{'='*60}")
    print(f"  Snowball: K=1->{K_MAX}, weight-tied, gated, local learning")
    print(f"{'='*60}")

    torch.manual_seed(42)
    embed = Embed(V, D, SEQ_LEN)
    block = TransformerBlock(D, D_FF, N_HEADS)
    readout = Readout(D, embed.tok.weight)
    model = Snowman(embed, block, readout, k_max=K_MAX)

    print(f"  params={model.param_count:,} "
          f"(block={model.block_params:,}, embed={model.embed_params:,}, head={model.readout_params:,})")
    r = flop_ratio(K_MAX, model.embed_params, model.block_params, model.readout_params)
    print(f"  FLOP ratio: {r:.3f}", flush=True)

    cfg = SnowballConfig(
        k_max=K_MAX, lr=LR, weight_decay=WD, grad_clip=1.0,
        eval_interval=EVAL_INTERVAL, log_interval=200,
        eval_batches=EVAL_BATCHES, seed=42,
        db_path=DB_PATH, run_name='snowball',
    )

    def on_step(step, K, loss, lr, flops):
        gv = model.gate_values[:K]
        gv_str = ','.join(f'{g:.3f}' for g in gv)
        if step % 200 == 0:
            print(f"  step {step:>5} | CE={loss:.4f} | K={K} | lr={lr:.2e} | a=[{gv_str}]",
                  flush=True)

    def on_eval(step, K, val_ce, flops):
        print(f"  >>> eval {step}: K{K}={val_ce:.4f}", flush=True)

    def on_phase(old_K, new_K, step, flops):
        print(f"  *** GROW: K={old_K}->{new_K} at step {step}", flush=True)

    trainer = SnowballTrainer(
        model, cfg, train_data, eval_data,
        tokens_per_step=BATCH * SEQ_LEN,
        on_step=on_step, on_eval=on_eval, on_phase=on_phase,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(flop_budget=flop_budget)
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        logger.log_summary('snowball', 'peak_vram_mb', f'{peak_mb:.1f}')
        logger.commit()
        print(f"  Snowball peak VRAM: {peak_mb:.0f}MB", flush=True)

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  RESULTS (FLOP-matched)")
    print(f"{'='*60}")
    for row in logger.conn.execute('SELECT run, key, value FROM summary ORDER BY run, key'):
        print(f"  {row[0]:>12} | {row[1]:>15} = {row[2]}")

    logger.close()


if __name__ == '__main__':
    main()
