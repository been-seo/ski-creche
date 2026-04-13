"""Width growth experiment: K=8 fixed, grow d only.

Train snowball K=1→8 at d=192, then grow width:
  d=192 (K=8) → d=384 (K=8) → d=768 (K=8)
Each width stage trains 1600 steps at fixed K=8.
Compare against baseline d=768 K=8 snowball from scratch.
"""
import math, sys, os, time, sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 256
BATCH = 16
D_HEAD = 64
LR = 3e-4
WD = 0.01
EVAL_INTERVAL = 200
EVAL_BATCHES = 10

# Phase 1: snowball K=1→8 at d=192, 200 steps/phase = 1600 steps
# Phase 2: d=384, K=8 fixed, 1600 steps
# Phase 3: d=768, K=8 fixed, 1600 steps
# Total: 4800 steps
# Baseline: d=768, K=1→8, 4800 steps (600/phase)

INIT_D = 192
INIT_HEADS = 3
K_MAX = 8
STEPS_PER_K_PHASE = 200   # snowball phase at initial d
STEPS_PER_WIDTH = 1600    # training after each width grow
WIDTHS = [(384, 6), (768, 12)]  # (d, n_heads) to grow into

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fineweb_elastic.db')


# ============================================================
#  Data
# ============================================================
class FineWebStream:
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

    def _fill(self, n):
        while len(self.buffer) < n:
            try:
                doc = next(self.iter)
                self.buffer.extend(self.enc.encode_ordinary(doc['text']))
            except StopIteration:
                self.iter = iter(self.ds)

    def get_batch(self):
        n = self.batch * (self.seq_len + 1)
        self._fill(n)
        t = torch.tensor(self.buffer[:n], dtype=torch.long).view(self.batch, self.seq_len + 1)
        self.buffer = self.buffer[n:]
        return t[:, :-1].to(DEVICE), t[:, 1:].to(DEVICE)


# ============================================================
#  Model (width-growable)
# ============================================================
class ActiveLayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d_active = d
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = 1e-5

    def forward(self, x):
        d = self.d_active
        a = x[..., :d]
        mu = a.mean(-1, keepdim=True)
        var = a.var(-1, keepdim=True, unbiased=False)
        normed = (a - mu) / torch.sqrt(var + self.eps)
        out = torch.zeros_like(x)
        out[..., :d] = self.weight[:d] * normed + self.bias[:d]
        return out

    def grow(self, new_d):
        old_d = self.weight.shape[0]
        if new_d <= old_d:
            self.d_active = new_d
            return
        new_w = torch.ones(new_d, device=self.weight.device)
        new_b = torch.zeros(new_d, device=self.weight.device)
        new_w[:old_d] = self.weight.data
        new_b[:old_d] = self.bias.data
        self.weight = nn.Parameter(new_w)
        self.bias = nn.Parameter(new_b)
        self.d_active = new_d


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.ln1 = ActiveLayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.ln2 = ActiveLayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, h):
        B, T, _ = h.shape
        nh, dh = self.n_heads, self.d_head
        h_n = self.ln1(h)
        Q = self.W_q(h_n).view(B, T, nh, dh).transpose(1, 2)
        K = self.W_k(h_n).view(B, T, nh, dh).transpose(1, 2)
        V = self.W_v(h_n).view(B, T, nh, dh).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.d)
        h = h + self.W_o(attn)
        h = h + self.ff2(F.gelu(self.ff1(self.ln2(h))))
        return h


def _pad_linear(layer, new_in, new_out):
    old_out, old_in = layer.weight.shape
    w = torch.zeros(new_out, new_in, device=layer.weight.device)
    w[:old_out, :old_in] = layer.weight.data
    layer.weight = nn.Parameter(w)
    if layer.bias is not None:
        b = torch.zeros(new_out, device=layer.bias.device)
        b[:old_out] = layer.bias.data
        layer.bias = nn.Parameter(b)
    layer.in_features = new_in
    layer.out_features = new_out


def grow_block(block, new_d, new_heads):
    old_d = block.d
    block.ln1.grow(new_d)
    block.ln2.grow(new_d)
    _pad_linear(block.W_q, new_d, new_d)
    _pad_linear(block.W_k, new_d, new_d)
    _pad_linear(block.W_v, new_d, new_d)
    _pad_linear(block.W_o, new_d, new_d)
    _pad_linear(block.ff1, new_d, 4 * new_d)
    _pad_linear(block.ff2, 4 * new_d, new_d)
    block.d = new_d
    block.n_heads = new_heads


class Embed(nn.Module):
    def __init__(self, vocab, d, seq_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.register_buffer('pe', self._make_pe(seq_len, d))
        self.d = d

    @staticmethod
    def _make_pe(seq_len, d):
        pe = torch.zeros(1, seq_len, d)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, idx):
        B, T = idx.shape
        return self.tok(idx) + self.pe[:, :T]

    def grow(self, new_d, seq_len):
        old_d = self.d
        V = self.tok.weight.shape[0]
        w = torch.zeros(V, new_d, device=self.tok.weight.device)
        w[:, :old_d] = self.tok.weight.data
        self.tok = nn.Embedding(V, new_d).to(w.device)
        self.tok.weight = nn.Parameter(w)
        self.pe = self._make_pe(seq_len, new_d).to(w.device)
        self.d = new_d


class Readout(nn.Module):
    def __init__(self, d, embed):
        super().__init__()
        self.ln = ActiveLayerNorm(d)
        self._embed = embed

    def forward(self, h):
        return F.linear(self.ln(h), self._embed.tok.weight)

    def grow(self, new_d):
        self.ln.grow(new_d)


class ElasticModel(nn.Module):
    def __init__(self, embed, block, readout, k_max):
        super().__init__()
        self.embed = embed
        self.block = block
        self.readout = readout
        self.gates = nn.ParameterList([
            nn.Parameter(torch.ones(1) if k == 0 else torch.zeros(1))
            for k in range(k_max)
        ])

    def forward(self, x, K):
        h = self.embed(x)
        for k in range(K):
            h_new = self.block(h)
            h = h + self.gates[k] * (h_new - h)
        return self.readout(h)

    def forward_single(self, x, K, targets):
        h = self.embed(x)
        if K > 1:
            with torch.no_grad():
                for k in range(K - 1):
                    h_new = self.block(h)
                    h = h + self.gates[k] * (h_new - h)
        h = h.detach()
        h_new = self.block(h)
        h = h + self.gates[K - 1] * (h_new - h)
        logits = self.readout(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        return loss.item()

    def grow_width(self, new_d, new_heads, seq_len):
        self.embed.grow(new_d, seq_len)
        grow_block(self.block, new_d, new_heads)
        self.readout.grow(new_d)

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def gate_values(self):
        return [g.item() for g in self.gates]


# ============================================================
#  Logger
# ============================================================
class Logger:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS log
            (run TEXT, step INT, d INT, K INT, ce REAL, val_ce REAL,
             lr REAL, params INT, timestamp REAL)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS summary
            (run TEXT, key TEXT, value TEXT, PRIMARY KEY(run, key))""")
        self.conn.commit()

    def log(self, run, step, d, K, ce, val_ce, lr, params):
        self.conn.execute("INSERT INTO log VALUES (?,?,?,?,?,?,?,?,?)",
                          (run, step, d, K, ce, val_ce, lr, params, time.time()))
        self.conn.commit()

    def summary(self, run, key, value):
        self.conn.execute("INSERT OR REPLACE INTO summary VALUES (?,?,?)",
                          (run, key, str(value)))
        self.conn.commit()

    def close(self):
        self.conn.close()


# ============================================================
#  Training helpers
# ============================================================
def evaluate(model, K, eval_data):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = eval_data.get_batch()
            logits = model(x, K)
            total += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total / EVAL_BATCHES


def train_phase(model, K, steps, train_data, eval_data, logger,
                run, global_step, d, label=""):
    """Train at fixed K for `steps` steps. Random depth sampling within 1..K."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_ce = float('inf')

    for i in range(steps):
        lr = LR * 0.5 * (1 + math.cos(math.pi * i / steps))
        for pg in opt.param_groups:
            pg['lr'] = lr

        x, y = train_data.get_batch()
        opt.zero_grad()
        k = torch.randint(1, K + 1, (1,)).item()
        ce = model.forward_single(x, k, y)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step = global_step + i
        if i % 200 == 0:
            gv = ','.join(f'{g:.3f}' for g in model.gate_values[:K])
            print(f"  {label}step {step:>5} | d={d} K={K} | CE={ce:.4f} | "
                  f"lr={lr:.2e} | a=[{gv}]", flush=True)

        if (i + 1) % EVAL_INTERVAL == 0:
            val = evaluate(model, K, eval_data)
            if val < best_ce:
                best_ce = val
            logger.log(run, step + 1, d, K, ce, val, lr, model.param_count)
            print(f"  {label}>>> eval {step+1}: d={d} K{K} val={val:.4f} "
                  f"(best={best_ce:.4f})", flush=True)

    return global_step + steps, best_ce


# ============================================================
#  Main runs
# ============================================================
def run_progressive(train_data, eval_data, logger):
    """K grows first at small d, then d grows at fixed K."""
    run = 'progressive'
    V = 50257

    print(f"\n{'='*60}")
    print(f"  Progressive: K=1->{K_MAX} at d={INIT_D}, then grow d")
    print(f"{'='*60}")

    torch.manual_seed(42)
    embed = Embed(V, INIT_D, SEQ_LEN)
    block = Block(INIT_D, INIT_HEADS)
    readout = Readout(INIT_D, embed)
    model = ElasticModel(embed, block, readout, k_max=K_MAX).to(DEVICE)

    global_step = 0
    t0 = time.time()
    vram_per_phase = {}

    # Phase 1: snowball K=1→K_MAX at d=INIT_D
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for K in range(1, K_MAX + 1):
        if K > 1:
            print(f"  *** GROW K: {K-1}->{K} at step {global_step}", flush=True)
        global_step, best = train_phase(
            model, K, STEPS_PER_K_PHASE, train_data, eval_data,
            logger, run, global_step, INIT_D)
    if torch.cuda.is_available():
        vram_per_phase[INIT_D] = torch.cuda.max_memory_allocated() / 1e6

    print(f"\n  Snowball done: d={INIT_D}, K={K_MAX}, "
          f"params={model.param_count:,}, step={global_step}, "
          f"VRAM={vram_per_phase.get(INIT_D, 0):.0f}MB", flush=True)

    # Phase 2+: grow width, train at K=K_MAX
    for new_d, new_nh in WIDTHS:
        old_d = model.block.d
        print(f"\n  *** WIDTH GROW: d={old_d} -> {new_d}, "
              f"heads={new_nh} at step {global_step} ***", flush=True)
        model.grow_width(new_d, new_nh, SEQ_LEN)
        # Move any new params to device
        model = model.to(DEVICE)
        print(f"  params={model.param_count:,}", flush=True)

        # Verify preservation
        val = evaluate(model, K_MAX, eval_data)
        print(f"  post-grow eval: K{K_MAX} val={val:.4f}", flush=True)

        # Reset VRAM tracking for this width phase
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        global_step, best = train_phase(
            model, K_MAX, STEPS_PER_WIDTH, train_data, eval_data,
            logger, run, global_step, new_d)

        if torch.cuda.is_available():
            vram_per_phase[new_d] = torch.cuda.max_memory_allocated() / 1e6

    elapsed = time.time() - t0
    final_ce = evaluate(model, K_MAX, eval_data)

    # Log per-phase VRAM
    for d_val, vram_val in vram_per_phase.items():
        logger.summary(run, f'vram_d{d_val}_mb', f'{vram_val:.1f}')
        print(f"  VRAM at d={d_val}: {vram_val:.0f} MB", flush=True)

    logger.summary(run, 'final_ce', f'{final_ce:.6f}')
    logger.summary(run, 'params', model.param_count)
    logger.summary(run, 'total_steps', global_step)
    logger.summary(run, 'elapsed', f'{elapsed:.1f}')
    logger.summary(run, 'peak_vram_mb', f'{vram_per_phase.get(WIDTHS[-1][0], 0):.1f}')
    print(f"\n  Progressive: CE={final_ce:.4f}, params={model.param_count:,}, "
          f"time={elapsed:.0f}s", flush=True)

    return model, global_step


def run_baseline(train_data, eval_data, logger, total_steps):
    """Baseline: d=768 K=1→8 snowball from scratch, same total steps."""
    run = 'baseline'
    V = 50257
    d, nh = WIDTHS[-1]

    print(f"\n{'='*60}")
    print(f"  Baseline: d={d} from scratch, {total_steps} total steps")
    print(f"{'='*60}")

    torch.manual_seed(42)
    embed = Embed(V, d, SEQ_LEN)
    block = Block(d, nh)
    readout = Readout(d, embed)
    model = ElasticModel(embed, block, readout, k_max=K_MAX).to(DEVICE)
    print(f"  params={model.param_count:,}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    steps_per_phase = total_steps // K_MAX
    global_step = 0
    t0 = time.time()

    for K in range(1, K_MAX + 1):
        if K > 1:
            print(f"  *** GROW K: {K-1}->{K} at step {global_step}", flush=True)
        global_step, best = train_phase(
            model, K, steps_per_phase, train_data, eval_data,
            logger, run, global_step, d, label="[base] ")

    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    final_ce = evaluate(model, K_MAX, eval_data)

    logger.summary(run, 'final_ce', f'{final_ce:.6f}')
    logger.summary(run, 'params', model.param_count)
    logger.summary(run, 'total_steps', global_step)
    logger.summary(run, 'elapsed', f'{elapsed:.1f}')
    logger.summary(run, 'peak_vram_mb', f'{vram:.1f}')
    logger.summary(run, 'vram_d768_mb', f'{vram:.1f}')
    print(f"\n  Baseline: CE={final_ce:.4f}, params={model.param_count:,}, "
          f"VRAM={vram:.0f}MB, time={elapsed:.0f}s", flush=True)

    return model


def depth_detach_test(model, eval_data, logger, run_name):
    """Evaluate at each K — depth detach test."""
    print(f"\n  Depth detach ({run_name}):")
    model.eval()
    for K in range(1, K_MAX + 1):
        total = 0.0
        with torch.no_grad():
            for _ in range(EVAL_BATCHES * 2):
                x, y = eval_data.get_batch()
                logits = model(x, K)
                total += F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)).item()
        ce = total / (EVAL_BATCHES * 2)
        print(f"    K={K}: CE={ce:.4f}")
        logger.summary(run_name, f'detach_K{K}', f'{ce:.4f}')
    model.train()


def main():
    rm_path = DB_PATH
    if os.path.exists(rm_path):
        os.remove(rm_path)
    logger = Logger(DB_PATH)

    train_data = FineWebStream(BATCH, SEQ_LEN, split='train', seed=0)
    eval_data = FineWebStream(BATCH, SEQ_LEN, split='train', seed=99999)

    # Progressive: K growth then d growth
    prog_model, total_steps = run_progressive(train_data, eval_data, logger)
    depth_detach_test(prog_model, eval_data, logger, 'progressive')
    del prog_model
    torch.cuda.empty_cache()

    # Baseline: d=768 from scratch, same steps
    base_model = run_baseline(train_data, eval_data, logger, total_steps)
    depth_detach_test(base_model, eval_data, logger, 'baseline')

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for row in logger.conn.execute(
            'SELECT run, key, value FROM summary ORDER BY run, key'):
        print(f"  {row[0]:>15} | {row[1]:>20} = {row[2]}")

    logger.close()


if __name__ == '__main__':
    main()
