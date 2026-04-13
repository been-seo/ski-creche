"""VRAM comparison: Snowball (per-depth backward) vs E2E."""
import sys, os, gc
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ski_creche import Block, Snowman

DEVICE = 'cuda'
SEQ_LEN = 256
BATCH = 32
D = 384
D_FF = 1536
N_HEADS = 6
VOCAB = 50257


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
    def __init__(self, vocab, d, seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
    def forward(self, idx):
        B, T = idx.shape
        return self.tok(idx) + self.pos(torch.arange(T, device=idx.device))


class Readout(nn.Module):
    def __init__(self, d, embed_weight):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        object.__setattr__(self, 'tied_weight', embed_weight)
    def forward(self, h):
        return F.linear(self.ln(h), self.tied_weight)


class E2EModel(nn.Module):
    def __init__(self, vocab, d, d_ff, n_heads, n_layers, seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, d_ff, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head_weight = self.tok.weight
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return F.linear(self.ln_f(x), self.head_weight)


def loss_fn(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def measure_vram(label, fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    peak = torch.cuda.max_memory_allocated()
    print(f"  {label}: peak={peak/1e6:.1f}MB (delta={( peak-base)/1e6:.1f}MB)")
    return peak


def main():
    x = torch.randint(VOCAB, (BATCH, SEQ_LEN), device=DEVICE)
    y = torch.randint(VOCAB, (BATCH, SEQ_LEN), device=DEVICE)

    for K in [2, 4, 8]:
        print(f"\n=== K={K} ===")

        # E2E
        gc.collect(); torch.cuda.empty_cache()
        e2e = E2EModel(VOCAB, D, D_FF, N_HEADS, K, SEQ_LEN).to(DEVICE)
        opt_e2e = torch.optim.AdamW(e2e.parameters(), lr=3e-4)

        def e2e_step():
            opt_e2e.zero_grad()
            logits = e2e(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt_e2e.step()

        peak_e2e = measure_vram(f"E2E (L={K})", e2e_step)

        del e2e, opt_e2e
        gc.collect(); torch.cuda.empty_cache()

        # Snowball
        embed = Embed(VOCAB, D, SEQ_LEN)
        block = TransformerBlock(D, D_FF, N_HEADS)
        readout = Readout(D, embed.tok.weight)
        model = Snowman(embed, block, readout, k_max=K).to(DEVICE)
        opt_snow = torch.optim.AdamW(model.parameters(), lr=3e-4)

        def snow_step():
            opt_snow.zero_grad()
            _, per_k = model.forward_local(x, K, y, loss_fn)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_snow.step()

        peak_snow = measure_vram(f"Snowball (K={K})", snow_step)

        print(f"  ratio: Snowball/E2E = {peak_snow/peak_e2e:.2f}x")

        del model, opt_snow
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
