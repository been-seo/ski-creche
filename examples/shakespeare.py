"""Example: Clay+Snowball on Shakespeare char-level LM."""
import sys, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from clay_snowball import (
    Block, DataStream, ClaySnowballConfig, ClayModel, SnowballTrainer, flop_ratio
)

DATA_PATH = os.path.join(os.path.dirname(__file__),
                         '..', 'experiments', 'data', 'shakespeare_input.txt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 128
BATCH = 64
D = 96
D_FF = 384
KERNEL = 7
K_MAX = 8


# ── Data ──────────────────────────────────────────────────────

class CharDataStream(DataStream):
    def __init__(self, data: np.ndarray, batch: int, seq_len: int, seed: int = 0):
        self.data = data
        self.batch = batch
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        self.n = len(data) - seq_len - 1

    def get_batch(self):
        idxs = [self.rng.randint(0, self.n) for _ in range(self.batch)]
        x = torch.tensor(np.stack([self.data[i:i+self.seq_len] for i in idxs]), dtype=torch.long)
        y = torch.tensor(np.stack([self.data[i+1:i+self.seq_len+1] for i in idxs]), dtype=torch.long)
        return x.to(DEVICE), y.to(DEVICE)


# ── Block ─────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x).transpose(1, 2)


class ConvBlock(Block):
    """CausalConv + FF with pre-norm residual."""
    def __init__(self, d, d_ff, kernel_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.conv = CausalConv1d(d, kernel_size)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)

    def forward(self, h):
        h = h + F.gelu(self.conv(self.ln1(h)))
        h = h + self.ff2(F.gelu(self.ff1(self.ln2(h))))
        return h


# ── Readout ───────────────────────────────────────────────────

class Readout(nn.Module):
    def __init__(self, d, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
    def forward(self, h):
        return self.head(self.ln(h))


# ── Embed ─────────────────────────────────────────────────────

class Embed(nn.Module):
    def __init__(self, vocab_size, d, seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(seq_len, d)
    def forward(self, idx):
        B, T = idx.shape
        return self.tok(idx) + self.pos(torch.arange(T, device=idx.device))


# ── Main ──────────────────────────────────────────────────────

def main():
    # Data
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    V = len(chars)
    n_val = int(len(data) * 0.1)
    train_data = CharDataStream(data[:-n_val], BATCH, SEQ_LEN, seed=0)
    eval_data = CharDataStream(data[-n_val:], BATCH, SEQ_LEN, seed=1)

    print(f"Vocab: {V}, Train: {len(data)-n_val:,}, Val: {n_val:,}")
    print(f"FLOP ratio (snowball avg / E2E): {flop_ratio(K_MAX, V*D+SEQ_LEN*D, sum(p.numel() for p in ConvBlock(D,D_FF,KERNEL).parameters()), V*D+D+D):.3f}")

    # Model
    embed = Embed(V, D, SEQ_LEN)
    block = ConvBlock(D, D_FF, KERNEL)
    readout = Readout(D, V)
    model = ClayModel(embed, block, readout, k_max=K_MAX)
    print(f"Params: {model.param_count:,} (block: {model.block_params:,})")

    # Config
    cfg = ClaySnowballConfig(
        k_max=K_MAX, lr=1e-3, weight_decay=0.01,
        total_steps=10000, eval_interval=1000, log_interval=500,
        device=DEVICE, seed=42,
    )

    # Callbacks
    def on_step(step, K, loss, lr, flops):
        gv = [f"{g:.2f}" for g in model.gate_values[:K]]
        print(f"  step {step:>5} | CE={loss:.4f} | K={K} | lr={lr:.2e} | α={gv}")

    def on_eval(step, K, metrics, flops):
        print(f"  >>> eval {step}: " + ", ".join(f"K{k}={v:.4f}" for k, v in sorted(metrics.items())))

    def on_phase(old_K, new_K, step, flops):
        print(f"  *** GROW: K={old_K}->{new_K} at step {step}")

    # Train
    trainer = SnowballTrainer(
        model, cfg, train_data, eval_data,
        tokens_per_step=BATCH * SEQ_LEN,
        on_step=on_step, on_eval=on_eval, on_phase=on_phase,
    )
    result = trainer.train()

    print(f"\nResult: best={result['best_loss']:.4f}, final={result['final_loss']:.4f}, "
          f"steps={result['total_steps']}, time={result['elapsed']:.0f}s")
    if result['final_metrics']:
        for k, v in sorted(result['final_metrics'].items()):
            print(f"  K={k}: {v:.4f}")


if __name__ == '__main__':
    main()
