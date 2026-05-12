"""
Adaptive progressive width growth for transformer language models.

Starts at d=2, grows on demand using a math-derived trigger that detects when
per-FLOP information gain drops below the eval-batch noise floor.

Trigger:  best progress in window W < k·σ_eval  (k=2, σ=0.05) → grow
Growth:   d → max(d+2, ceil(d·α/2)·2),  α=1.05 (multiplicative, even-rounded)
Recovery: W steps after grow with no trigger check
Init:     mean-of-existing weight rows/cols + 2%·std Gaussian noise
          (smoother than zero-pad, breaks symmetry across new dims)

Caveat: zero-pad with standard LayerNorm causes a one-time mean/var shift on
each grow.  Mean-init reduces the shock per grow but accumulates a small drift
across many grows.  See README for tradeoffs and the ActiveLayerNorm
alternative from proofs/width_elasticity.py.

Usage:
    CACHE_PATH=/path/to/fineweb_tokens.pt python adaptive_growth.py
"""
import math, os, queue, sqlite3, threading, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = 'cuda'
DTYPE = torch.bfloat16
V = 50257
D_HEAD = 2
N_LAYERS = 22
BATCH, SEQ_LEN = 64, 512
LR, WD, GRAD_CLIP = 3e-4, 0.01, 1.0
USE_8BIT = True
D_MAX = 2048
GROWTH_FACTOR = 1.05  # multiplicative growth (5% more dim per grow)
GROWTH_MIN_STEP = 2   # absolute floor for small d
WINDOW_STEPS = 2000   # progress measurement window
SIGMA_EVAL = 0.05     # eval-batch noise floor (std of val CE across batches)
NOISE_K = 2.0         # threshold = NOISE_K · SIGMA_EVAL
PROGRESS_THRESHOLD = NOISE_K * SIGMA_EVAL  # 0.10 default
VRAM_CAP_RATIO = 0.90  # stop growing when VRAM > this fraction

CACHE_PATH = os.environ.get('CACHE_PATH', './data_cache/fineweb_2.9B.pt')
EVAL_TOKENS = 1_000_000
DB_PATH = os.environ.get('DB_PATH', './adaptive.db')
CKPT_PREFIX = 'adap_'
CKPT_DIR = os.environ.get('CKPT_DIR', './ckpts')
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LOG_INTERVAL = 100

# FLOP budget
TARGET_FLOPS = 6 * (V * 2048 + N_LAYERS * 12 * 2048 * 2048) * BATCH * SEQ_LEN * 25000

def flops_per_step(d):
    return 6 * (V * d + N_LAYERS * 12 * d * d) * BATCH * SEQ_LEN

# ── data ──
class DataLoader:
    def __init__(self, split='train', seed=0):
        print(f'  [data] loading...', flush=True)
        all_tokens = torch.load(CACHE_PATH, weights_only=True).long()
        self._tokens = all_tokens[:-EVAL_TOKENS] if split == 'train' else all_tokens[-EVAL_TOKENS:]
        print(f'  [data] {split}: {self._tokens.shape[0]/1e6:.0f}M tokens', flush=True)
        self._pos = (seed * BATCH * (SEQ_LEN + 1)) % self._tokens.shape[0]
        self._q = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()
    def _worker(self):
        n = BATCH * (SEQ_LEN + 1)
        while not self._stop.is_set():
            if self._pos + n > self._tokens.shape[0]: self._pos = 0
            t = self._tokens[self._pos:self._pos+n].view(BATCH, SEQ_LEN+1)
            self._pos += n
            self._q.put((t[:,:-1], t[:,1:]))
    def get_batch(self):
        x, y = self._q.get()
        return x.to(DEVICE), y.to(DEVICE)
    def close(self): self._stop.set()

# ── model ──
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.n_heads = max(1, d // D_HEAD)
        self.d_head = d // self.n_heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.ff1 = nn.Linear(d, 4*d); self.ff2 = nn.Linear(4*d, d)
    def forward(self, h):
        B, T, _ = h.shape
        nh, dh = self.n_heads, self.d_head
        h_n = self.ln1(h)
        q = self.W_q(h_n).view(B,T,nh,dh).transpose(1,2)
        k = self.W_k(h_n).view(B,T,nh,dh).transpose(1,2)
        v = self.W_v(h_n).view(B,T,nh,dh).transpose(1,2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.W_o(a.transpose(1,2).contiguous().view(B,T,self.d))
        h = h + self.ff2(F.gelu(self.ff1(self.ln2(h))))
        return h

def _grow_linear(layer, new_in, new_out, noise=0.02):
    """Mean-init growth: new rows/cols = mean of existing + small Gaussian noise.
    Smoother than zero-pad: LN statistics roughly preserved, gradient signal flows.
    Symmetry breaking via small noise (relative to weight std).
    """
    old_w = layer.weight.data
    old_out, old_in = old_w.shape
    dev = old_w.device
    w = torch.empty(new_out, new_in, device=dev)
    w[:old_out, :old_in] = old_w
    n_extra_out = new_out - old_out
    n_extra_in = new_in - old_in
    if old_w.numel() > 0:
        col_mean = old_w.mean(dim=0, keepdim=True)   # (1, old_in)
        row_mean = old_w.mean(dim=1, keepdim=True)   # (old_out, 1)
        global_mean = old_w.mean()
        w_std = old_w.std() if old_w.numel() > 1 else torch.tensor(0.02, device=dev)
    else:
        col_mean = torch.zeros(1, old_in, device=dev)
        row_mean = torch.zeros(old_out, 1, device=dev)
        global_mean = torch.tensor(0.0, device=dev)
        w_std = torch.tensor(0.02, device=dev)
    if n_extra_out > 0 and old_in > 0:
        w[old_out:, :old_in] = col_mean.expand(n_extra_out, -1)
    if n_extra_in > 0 and old_out > 0:
        w[:old_out, old_in:] = row_mean.expand(-1, n_extra_in)
    if n_extra_out > 0 and n_extra_in > 0:
        w[old_out:, old_in:] = global_mean
    if noise > 0 and (n_extra_out > 0 or n_extra_in > 0):
        sigma = noise * w_std.item()
        if n_extra_out > 0:
            w[old_out:, :] += torch.randn(n_extra_out, new_in, device=dev) * sigma
        if n_extra_in > 0:
            w[:old_out, old_in:] += torch.randn(old_out, n_extra_in, device=dev) * sigma
    layer.weight = nn.Parameter(w)
    if layer.bias is not None:
        old_b = layer.bias.data
        new_b = torch.empty(new_out, device=dev)
        new_b[:old_out] = old_b
        if new_out > old_out:
            b_mean = old_b.mean() if old_b.numel() > 0 else torch.tensor(0.0, device=dev)
            b_std = old_b.std() if old_b.numel() > 1 else torch.tensor(0.02, device=dev)
            new_b[old_out:] = b_mean + torch.randn(n_extra_out, device=dev) * (noise * b_std.item())
        layer.bias = nn.Parameter(new_b)
    layer.in_features, layer.out_features = new_in, new_out

def _grow_layernorm(ln, new_d):
    """LN: new gain = mean(old gain) ≈ 1; new bias = mean(old bias) ≈ 0."""
    old_d = ln.weight.shape[0]
    dev = ln.weight.device
    w = torch.empty(new_d, device=dev)
    b = torch.empty(new_d, device=dev)
    w[:old_d] = ln.weight.data; b[:old_d] = ln.bias.data
    if new_d > old_d:
        w[old_d:] = ln.weight.data.mean()
        b[old_d:] = ln.bias.data.mean()
    ln.weight = nn.Parameter(w); ln.bias = nn.Parameter(b)
    ln.normalized_shape = (new_d,)

def grow_block(block, new_d):
    old_d = block.d
    _grow_layernorm(block.ln1, new_d)
    _grow_layernorm(block.ln2, new_d)
    for lyr in (block.W_q, block.W_k, block.W_v, block.W_o):
        _grow_linear(lyr, new_d, new_d)
    _grow_linear(block.ff1, new_d, 4*new_d)
    _grow_linear(block.ff2, 4*new_d, new_d)
    block.d = new_d
    block.n_heads = max(1, new_d // D_HEAD)
    block.d_head = new_d // block.n_heads

class Model(nn.Module):
    def __init__(self, d, n_layers, seq_len):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(V, d)
        self.register_buffer('pe', self._sinpe(seq_len, d))
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.embed.weight
    @staticmethod
    def _sinpe(seq_len, d):
        pe = torch.zeros(1, seq_len, d)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / max(d,2)))
        pe[0,:,0::2] = torch.sin(pos * div[:d//2+d%2])
        pe[0,:,1::2] = torch.cos(pos * div[:d//2])
        return pe
    def forward(self, idx):
        h = self.embed(idx) + self.pe[:,:idx.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))
    def forward_with_hidden(self, idx):
        """Forward pass returning final hidden states for d_eff measurement."""
        h = self.embed(idx) + self.pe[:,:idx.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h)), h
    def grow(self, new_d):
        old_d = self.d; dev = self.embed.weight.device
        # Embed: new cols = mean of existing cols + noise
        old_emb = self.embed.weight.data
        w = torch.empty(V, new_d, device=dev)
        w[:, :old_d] = old_emb
        if new_d > old_d:
            row_mean = old_emb.mean(dim=1, keepdim=True)
            emb_std = old_emb.std()
            w[:, old_d:] = row_mean.expand(-1, new_d - old_d)
            w[:, old_d:] += torch.randn(V, new_d - old_d, device=dev) * (0.02 * emb_std.item())
        self.embed = nn.Embedding(V, new_d).to(dev)
        self.embed.weight = nn.Parameter(w)
        # PE: extend with fresh sinusoid for new dims (preserves orthogonal structure)
        new_pe = self._sinpe(SEQ_LEN, new_d).to(dev)
        new_pe[:,:,:old_d] = self.pe[:,:SEQ_LEN,:old_d]
        self.pe = new_pe
        for block in self.blocks:
            grow_block(block, new_d)
        _grow_layernorm(self.ln_f, new_d)
        # head shares embed weight; just update Linear metadata
        self.head = nn.Linear(new_d, V, bias=False).to(dev)
        self.head.weight = self.embed.weight
        self.d = new_d
    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

@torch.no_grad()
def measure_d_eff(model, data):
    """Measure effective rank of hidden states via SVD."""
    model.eval()
    x, _ = data.get_batch()
    with torch.autocast(device_type='cuda', dtype=DTYPE):
        _, h = model.forward_with_hidden(x)
    # h: (B, T, d) → flatten to (B*T, d)
    h_flat = h.float().reshape(-1, model.d)
    # Sample subset for efficiency
    if h_flat.shape[0] > 4096:
        idx = torch.randperm(h_flat.shape[0])[:4096]
        h_flat = h_flat[idx]
    # SVD
    _, s, _ = torch.linalg.svd(h_flat, full_matrices=False)
    # Effective rank: exp(entropy of normalized singular values)
    s = s / s.sum()
    s = s[s > 1e-10]  # remove zeros
    entropy = -(s * s.log()).sum()
    d_eff = entropy.exp().item()
    model.train()
    return d_eff

# ── logger ──
class Logger:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS log "
            "(run TEXT, step INT, d INT, d_eff REAL, ce REAL, val_ce REAL, "
            "lr REAL, params INT, vram_mb REAL, ts REAL, flops_used REAL)")
        self.conn.commit()
    def log(self, run, step, d, d_eff, ce, val_ce, lr, params, flops_used):
        vram = torch.cuda.max_memory_allocated() / 1e6
        self.conn.execute(
            "INSERT INTO log VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run, step, d, d_eff, ce, val_ce or 0, lr, params, vram,
             time.time(), flops_used))
        self.conn.commit()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', default=None)
    args = p.parse_args()

    print(f'device={DEVICE}  batch={BATCH}  seq={SEQ_LEN}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Target FLOPs: {TARGET_FLOPS:.2e}')
    print(f'D_HEAD={D_HEAD}, max d={D_MAX}')

    train_data = DataLoader('train', seed=0)
    eval_data = DataLoader('train', seed=99999)
    logger = Logger(DB_PATH)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model = Model(ckpt['d'], N_LAYERS, SEQ_LEN).to(DEVICE)
        model.load_state_dict(ckpt['model'])
        global_step = ckpt['step']
        flops_used = ckpt.get('flops_used', 0)
        best = ckpt.get('best', float('inf'))
        print(f'  Resumed: step={global_step}, d={ckpt["d"]}, flops={flops_used:.2e}')
    else:
        model = Model(2, N_LAYERS, SEQ_LEN).to(DEVICE)
        global_step = 0
        flops_used = 0
        best = float('inf')

    print(f'Model: d={model.d}, {N_LAYERS}L, {model.param_count/1e6:.2f}M params')

    if USE_8BIT:
        import bitsandbytes as bnb
        make_opt = lambda m: bnb.optim.AdamW8bit(m.parameters(), lr=LR, weight_decay=WD)
    else:
        make_opt = lambda m: torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)

    opt = make_opt(model)
    t0 = time.time()
    run = 'adaptive_v3'
    last_d_eff = 0
    last_growth_step = -WINDOW_STEPS  # allow trigger from the start
    # val history at current d (reset on grow); list of (step, val)
    val_history = []
    eval_interval_count = WINDOW_STEPS // EVAL_INTERVAL  # = 4

    print(f'\n{"="*60}')
    print(f'  Adaptive Growth v3 (math-derived):')
    print(f'    window={WINDOW_STEPS} step ({eval_interval_count} evals)')
    print(f'    threshold={PROGRESS_THRESHOLD:.3f} (= {NOISE_K}·σ_eval={SIGMA_EVAL})')
    print(f'    growth: d → max(d+{GROWTH_MIN_STEP}, ⌈d·{GROWTH_FACTOR}/2⌉·2), max={D_MAX}')
    print(f'{"="*60}')

    while flops_used < TARGET_FLOPS and model.d <= D_MAX:
        # cosine LR based on remaining FLOP fraction
        frac = flops_used / TARGET_FLOPS
        lr = LR * 0.5 * (1 + math.cos(math.pi * frac))
        for pg in opt.param_groups: pg['lr'] = lr

        x, y = train_data.get_batch()
        opt.zero_grad()
        with torch.autocast(device_type='cuda', dtype=DTYPE):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        step_flops = flops_per_step(model.d)
        flops_used += step_flops
        ce = loss.item()

        if global_step % LOG_INTERVAL == 0:
            print(f'  step {global_step:>6} d={model.d} CE={ce:.4f} lr={lr:.2e} '
                  f'FLOP={flops_used/TARGET_FLOPS*100:.1f}%', flush=True)

        # (activation d_eff trigger disabled — PE-contaminated, see analysis)
        # d_eff diagnostic measurement only
        if (global_step + 1) % 1000 == 0:
            last_d_eff = measure_d_eff(model, train_data)

        # eval
        if (global_step + 1) % EVAL_INTERVAL == 0:
            model.eval()
            total = 0
            with torch.no_grad():
                for _ in range(EVAL_BATCHES):
                    x, y = eval_data.get_batch()
                    total += F.cross_entropy(
                        model(x).view(-1, V), y.view(-1)).item()
            val = total / EVAL_BATCHES
            in_recovery = (global_step - last_growth_step) < WINDOW_STEPS
            if val < best:
                best = val
            val_history.append((global_step + 1, val))
            model.train()
            logger.log(run, global_step+1, model.d, last_d_eff, ce, val, lr,
                       model.param_count, flops_used)

            # Math-derived trigger: progress in last W < threshold
            # Compute best_W_ago vs best_recent (each over W/2 window)
            half = eval_interval_count  # use full W for reliability vs half
            progress = None
            if not in_recovery and len(val_history) >= 2 * half:
                old_window = val_history[-2*half:-half]  # W steps ago
                new_window = val_history[-half:]         # last W steps
                best_old = min(v for _, v in old_window)
                best_new = min(v for _, v in new_window)
                progress = best_old - best_new  # positive if improving

            # VRAM utilization (capacity = total GPU memory)
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_used = torch.cuda.memory_allocated()
            vram_ratio = vram_used / vram_total
            vram_capped = vram_ratio > VRAM_CAP_RATIO

            recov_tag = ' [recov]' if in_recovery else ''
            cap_tag = ' [VRAM-CAP]' if vram_capped else ''
            prog_str = f'prog={progress:+.3f}' if progress is not None else 'prog=NA'
            print(f'  >>> eval {global_step+1} d={model.d} val={val:.4f} '
                  f'best={best:.4f} {prog_str}{recov_tag}{cap_tag} '
                  f'vram={vram_ratio*100:.0f}% '
                  f'FLOP={flops_used/TARGET_FLOPS*100:.1f}%', flush=True)

            if (progress is not None and progress < PROGRESS_THRESHOLD
                    and model.d < D_MAX and not vram_capped):
                # multiplicative growth, even-rounded
                target = math.ceil(model.d * GROWTH_FACTOR / 2) * 2
                new_d = min(max(model.d + GROWTH_MIN_STEP, target), D_MAX)
                print(f'  *** GROW d={model.d}→{new_d} '
                      f'(progress={progress:+.3f} < {PROGRESS_THRESHOLD}) ***',
                      flush=True)
                model.grow(new_d)
                model = model.to(DEVICE)
                opt = make_opt(model)
                last_growth_step = global_step
                val_history = []  # reset history at new d
                print(f'  params={model.param_count/1e6:.2f}M', flush=True)

        # checkpoint
        if (global_step + 1) % 5000 == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            path = os.path.join(CKPT_DIR, f'{CKPT_PREFIX}step{global_step+1}.pt')
            torch.save({'model': model.state_dict(), 'step': global_step+1,
                        'd': model.d, 'flops_used': flops_used, 'best': best},
                       path)
            print(f'  [ckpt] saved {path}', flush=True)

        global_step += 1

    # final
    elapsed = time.time() - t0
    model.eval()
    total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = eval_data.get_batch()
            total += F.cross_entropy(model(x).view(-1, V), y.view(-1)).item()
    final = total / EVAL_BATCHES

    print(f'\n{"="*60}')
    print(f'  DONE: val={final:.4f} best={best:.4f}')
    print(f'  final d={model.d}, params={model.param_count/1e9:.2f}B')
    print(f'  FLOPs used: {flops_used:.2e} ({flops_used/TARGET_FLOPS*100:.1f}%)')
    print(f'  time={elapsed:.0f}s')
    print(f'  E2E fixed: val=4.97')
    print(f'{"="*60}')

    train_data.close(); eval_data.close()

if __name__ == '__main__':
    main()
