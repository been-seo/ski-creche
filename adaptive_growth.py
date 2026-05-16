"""
Adaptive progressive width growth for Snowman language models.

Function-preserving grow + LN d_active ramp (math-derived).

At grow event d_old → d_new, |ΔL_grow|=0 EXACT iff:
  (I1) residual stream new dims ≡ 0 forever
       (W_o, ff2 new rows = 0; embed/PE new cols = 0)
  (I2) LN statistics unchanged at grow
       (d_active stays at d_old at grow)
  (I3) old residual unaffected by new attention heads / FFN hidden units
       (W_o, ff2 new INPUT cols = 0)

Post-grow activation via smooth fractional d_active ramp.  Cauchy-Schwarz:
  per-step |ΔL| ≤ |∇L| · N_layers · ‖∂Y/∂d_active‖ · Δd_active.
With Δd_active/step = 1/K_PER_DIM, per-step ΔL is bounded.
K_PER_DIM ≥ N_layers/(α·σ_eval·√n) ≈ 22/(0.5·0.05·√8) ≈ 200.

Trigger: stalled (best progress in window W < k·σ_eval) AND ramp complete.
No grow mid-ramp.

Environment variables (all optional):
  CACHE_PATH, DB_PATH, CKPT_DIR, RUN_NAME, START_D, SEED, TARGET_FLOPS.
"""
import math, os, queue, sqlite3, threading, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = 'cuda'
DTYPE = torch.bfloat16
V = 50257
D_HEAD = 2  # minimum d_head; actual d_head grows with d via d_head_for(d)
N_LAYERS = 22

def d_head_for(d):
    """d_head ramps up with d to keep n_heads bounded."""
    if d <= 8:    return 2
    if d <= 32:   return 4
    if d <= 128:  return 8
    if d <= 512:  return 16
    return 32

def round_d_to_valid(target_d):
    """Round target_d up to nearest multiple of d_head_for(target_d)."""
    dh = d_head_for(target_d)
    return ((target_d + dh - 1) // dh) * dh

BATCH, SEQ_LEN = 64, 512
LR, WD, GRAD_CLIP = 1.5e-4, 0.01, 1.0
USE_8BIT = False
USE_AUTOCAST = True
D_MAX = 2048
GROWTH_FACTOR = 1.05
GROWTH_MIN_STEP = 2
WINDOW_STEPS = 2000
SIGMA_EVAL = 0.05
NOISE_K = 2.0
PROGRESS_THRESHOLD = NOISE_K * SIGMA_EVAL
ROW_NOISE = 1e-3
VRAM_CAP_RATIO = 0.90
K_PER_DIM = 200  # steps per d_active increment during post-grow ramp.
                  # Derived: K_PER_DIM ≥ N_layers/(α·σ_eval·√n) ≈ 200 at n=8.

CACHE_PATH = os.environ.get('CACHE_PATH', 'data_cache/fineweb_2.9B.pt')
EVAL_TOKENS = 1_000_000
DB_PATH = os.environ.get('DB_PATH', 'adaptive.db')
CKPT_PREFIX = 'adaptive_'
CKPT_DIR = os.environ.get('CKPT_DIR', 'ckpts')
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LOG_INTERVAL = 100

TARGET_FLOPS = float(os.environ.get(
    'TARGET_FLOPS',
    6 * (V * 2048 + N_LAYERS * 12 * 2048 * 2048) * BATCH * SEQ_LEN * 25000))

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
class ActiveLayerNorm(nn.Module):
    """LayerNorm with FRACTIONAL active width (PAVING width-elasticity).

    d_active ∈ ℝ⁺.  Weighted mean/var:
       w_i = 1 for i < floor(d_a), w_i = frac for i = floor(d_a), w_i = 0 above
       mean = Σ w_i x_i / Σ w_i,   var = Σ w_i (x_i - mean)² / Σ w_i
       LN_out_i = w_i · (gain_i · (x_i - mean) / √(var+ε) + bias_i)

    Smoothness: dLN/d_active is continuous and bounded for all d_active ∈ [1, D]
    ⇒ per-step ΔL ≤ |∇L|·N_layers·‖∂Y/∂d_active‖·Δd_active.  With ramp rate
    Δd_active/step = 1/K_PER_DIM, per-step ΔL is bounded by Cauchy-Schwarz."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.d_active = float(d)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        D = x.shape[-1]
        n_int = int(self.d_active)
        frac = self.d_active - n_int
        w = torch.zeros(D, device=x.device, dtype=x.dtype)
        if n_int >= D:
            w[:] = 1.0
        else:
            if n_int > 0:
                w[:n_int] = 1.0
            if frac > 0.0:
                w[n_int] = frac
        n_eff = w.sum()
        if n_eff < 1e-6:
            return torch.zeros_like(x)
        wx = w * x
        mu = wx.sum(dim=-1, keepdim=True) / n_eff
        cent = x - mu
        var = (w * cent * cent).sum(dim=-1, keepdim=True) / n_eff
        normed = cent / torch.sqrt(var + self.eps)
        out = w * (self.weight * normed + self.bias)
        return out

GATE_EPS = 1e-3

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.d_head = d_head_for(d)
        self.n_heads = d // self.d_head
        self.ln1 = ActiveLayerNorm(d); self.ln2 = ActiveLayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.ff1 = nn.Linear(d, 4*d); self.ff2 = nn.Linear(4*d, d)
        # Per-output-dim gates on q/k/v/ff1.  Init=1; at grow new dims get
        # GATE_EPS so new-head contribution is gate-suppressed at grow,
        # ramps up via Adam learning the gate.
        self.gate_q = nn.Parameter(torch.ones(d))
        self.gate_k = nn.Parameter(torch.ones(d))
        self.gate_v = nn.Parameter(torch.ones(d))
        self.gate_ff1 = nn.Parameter(torch.ones(4*d))
    def forward(self, h):
        B, T, _ = h.shape
        nh, dh = self.n_heads, self.d_head
        h_n = self.ln1(h)
        q = (self.W_q(h_n) * self.gate_q).view(B,T,nh,dh).transpose(1,2)
        k = (self.W_k(h_n) * self.gate_k).view(B,T,nh,dh).transpose(1,2)
        v = (self.W_v(h_n) * self.gate_v).view(B,T,nh,dh).transpose(1,2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.W_o(a.transpose(1,2).contiguous().view(B,T,self.d))
        h = h + self.ff2(F.gelu(self.ff1(self.ln2(h)) * self.gate_ff1))
        return h

def _grow_linear(layer, new_in, new_out, noise_new_rows=0.0):
    """Zero-pad growth with optional small noise on NEW ROWS only.

    Why asymmetric:
      Pure zero-pad + ActiveLayerNorm makes growth function-preserving on its
      own, but new attention heads' q-row stays at 0 forever:
        q[new_row] = W_q[new_row,:] @ ln_x = 0  →  attn out=0  →  no
        contribution to loss  →  ∂L/∂W_q[new_row,:] = 0.
      Tiny noise on NEW ROWS of W_q,W_k,W_v breaks this dead loop: q,k,v at
      new heads become nonzero, and W_o[:, new_cols] (kept at 0 → function
      preserved) starts receiving real gradient.  Once W_o new cols learn,
      gradient flows back into the new q/k/v rows and the head trains.
    """
    old_w = layer.weight.data
    old_out, old_in = old_w.shape
    dev = old_w.device
    w = torch.zeros(new_out, new_in, device=dev)
    w[:old_out, :old_in] = old_w
    if noise_new_rows > 0 and new_out > old_out:
        scale = noise_new_rows
        if old_w.numel() > 1:
            scale = noise_new_rows * old_w.std().item()
        w[old_out:, :old_in] = torch.randn(new_out - old_out, old_in, device=dev) * scale
    layer.weight = nn.Parameter(w)
    if layer.bias is not None:
        b = torch.zeros(new_out, device=dev)
        b[:old_out] = layer.bias.data
        layer.bias = nn.Parameter(b)
    layer.in_features, layer.out_features = new_in, new_out

def _grow_active_ln(ln, new_d, activate=False):
    """Extend gain/bias to new_d (gain=1, bias=0 for new entries).
    activate=False (default): d_active stays at old.  The training loop ramps
    d_active to new_d gradually via Model.step_ramp."""
    old_d = ln.weight.shape[0]
    dev = ln.weight.device
    w = torch.ones(new_d, device=dev)
    b = torch.zeros(new_d, device=dev)
    w[:old_d] = ln.weight.data; b[:old_d] = ln.bias.data
    ln.weight = nn.Parameter(w); ln.bias = nn.Parameter(b)
    if activate:
        ln.d_active = new_d

def _grow_gate(block, name, new_size):
    """Extend a learnable gate parameter, init new entries to GATE_EPS."""
    old = getattr(block, name).data
    dev = old.device
    new = torch.full((new_size,), GATE_EPS, device=dev)
    new[:old.shape[0]] = old
    setattr(block, name, nn.Parameter(new))

def grow_block(block, new_d):
    """Asymmetric grow: noise on new rows of q/k/v/ff1 (so gradient flows in),
    zero on new rows of W_o/ff2 (so residual stream stays function-preserved),
    GATE_EPS on new gate entries (smooth contribution ramp)."""
    _grow_active_ln(block.ln1, new_d)
    _grow_active_ln(block.ln2, new_d)
    _grow_linear(block.W_q, new_d, new_d, noise_new_rows=ROW_NOISE)
    _grow_linear(block.W_k, new_d, new_d, noise_new_rows=ROW_NOISE)
    _grow_linear(block.W_v, new_d, new_d, noise_new_rows=ROW_NOISE)
    _grow_linear(block.W_o, new_d, new_d, noise_new_rows=0.0)
    _grow_linear(block.ff1, new_d, 4*new_d, noise_new_rows=ROW_NOISE)
    _grow_linear(block.ff2, 4*new_d, new_d, noise_new_rows=0.0)
    _grow_gate(block, 'gate_q', new_d)
    _grow_gate(block, 'gate_k', new_d)
    _grow_gate(block, 'gate_v', new_d)
    _grow_gate(block, 'gate_ff1', 4*new_d)
    block.d = new_d
    block.d_head = d_head_for(new_d)
    block.n_heads = new_d // block.d_head

class Model(nn.Module):
    def __init__(self, d, n_layers, seq_len):
        super().__init__()
        self.d = d
        self.d_active_target = d   # ramp target; equals d when not in ramp
        self.last_ramp_step = 0
        self.embed = nn.Embedding(V, d)
        self.register_buffer('pe', self._sinpe(seq_len, d))
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.ln_f = ActiveLayerNorm(d)
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
        h = self.embed(idx) + self.pe[:,:idx.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h)), h
    def grow(self, new_d):
        """Function-preserving grow: d_active stays at old_d for all LNs.
        d_active_target = new_d signals ramp in progress."""
        old_d = self.d; dev = self.embed.weight.device
        old_emb = self.embed.weight.data
        w = torch.zeros(V, new_d, device=dev)
        w[:, :old_d] = old_emb
        self.embed = nn.Embedding(V, new_d).to(dev)
        self.embed.weight = nn.Parameter(w)
        new_pe = torch.zeros(1, SEQ_LEN, new_d, device=dev)
        new_pe[:,:,:old_d] = self.pe[:,:SEQ_LEN,:old_d]
        self.pe = new_pe
        for block in self.blocks:
            grow_block(block, new_d)
        _grow_active_ln(self.ln_f, new_d, activate=False)
        self.head = nn.Linear(new_d, V, bias=False).to(dev)
        self.head.weight = self.embed.weight
        self.d = new_d
        self.d_active_target = new_d

    def step_ramp(self, current_step, k_per_dim):
        """Smooth ramp: advance d_active by Δ = 1/k_per_dim each step until target.
        Cauchy-Schwarz: per-step ΔL ≤ |∇L|·‖dY/dλ‖/k_per_dim."""
        d_active = self.ln_f.d_active
        if d_active >= self.d_active_target:
            return d_active
        new_active = min(d_active + 1.0 / k_per_dim, float(self.d_active_target))
        for block in self.blocks:
            block.ln1.d_active = new_active
            block.ln2.d_active = new_active
        self.ln_f.d_active = new_active
        self.last_ramp_step = current_step
        return new_active

    @property
    def d_active(self):
        return self.ln_f.d_active

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

@torch.no_grad()
def measure_d_eff(model, data=None):
    """Effective rank of activation matrix (residual stream after final ln).
    H ∈ R^{B·T × d} → r_eff(H) via entropy of normalized singular values.
    Returns r_eff / d in [1/d, 1]."""
    model.eval()
    with torch.no_grad():
        if data is None:
            x = torch.randint(0, V, (16, SEQ_LEN), device=DEVICE)
        else:
            x, _ = data.get_batch()
            x = x[:16]
        if USE_AUTOCAST:
            with torch.autocast(device_type='cuda', dtype=DTYPE):
                _, h = model.forward_with_hidden(x)
        else:
            _, h = model.forward_with_hidden(x)
        h = h.reshape(-1, model.d).float()
        cov = h.T @ h / h.shape[0]
        eigs = torch.linalg.eigvalsh(cov)
        eigs = eigs.clamp(min=0)
        s = eigs.sqrt()
        s_sum = s.sum()
        if s_sum < 1e-10:
            return 0.0
        p = s / s_sum
        p = p[p > 1e-10]
        entropy = -(p * p.log()).sum()
        r_eff = entropy.exp().item()
    model.train()
    return r_eff / model.d

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

    SEED = int(os.environ.get('SEED', 0))
    torch.manual_seed(SEED)
    train_data = DataLoader('train', seed=SEED)
    eval_data = DataLoader('train', seed=99999)
    logger = Logger(DB_PATH)

    wall_offset = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model = Model(ckpt['d'], N_LAYERS, SEQ_LEN).to(DEVICE)
        model.load_state_dict(ckpt['model'])
        global_step = ckpt['step']
        flops_used = ckpt.get('flops_used', 0)
        best = ckpt.get('best', float('inf'))
        wall_offset = ckpt.get('wall', 0.0)
        wall_offset = float(os.environ.get('WALL_OFFSET', wall_offset))
        print(f'  Resumed: step={global_step}, d={ckpt["d"]}, flops={flops_used:.2e}, wall_offset={wall_offset/3600:.2f}h')
    else:
        start_d = int(os.environ.get('START_D', 2))
        model = Model(start_d, N_LAYERS, SEQ_LEN).to(DEVICE)
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
    t0 = time.time() - wall_offset
    run = os.environ.get('RUN_NAME', 'adaptive')
    last_d_eff = 0
    last_growth_step = -WINDOW_STEPS
    val_history = []
    eval_interval_count = WINDOW_STEPS // EVAL_INTERVAL

    print(f'\n{"="*60}')
    print(f'  Adaptive Growth (math-derived):')
    print(f'    window={WINDOW_STEPS} step ({eval_interval_count} evals)')
    print(f'    threshold={PROGRESS_THRESHOLD:.3f} (= {NOISE_K}·σ_eval={SIGMA_EVAL})')
    print(f'    growth: d → max(d+{GROWTH_MIN_STEP}, ⌈d·{GROWTH_FACTOR}/2⌉·2), max={D_MAX}')
    print(f'    ramp:   {K_PER_DIM} steps per d_active increment')
    print(f'{"="*60}')

    while flops_used < TARGET_FLOPS and model.d <= D_MAX:
        # Ramp d_active smoothly toward target (function-preserving init →
        # live capacity via Cauchy-Schwarz-bounded per-step ΔL).
        model.step_ramp(global_step, K_PER_DIM)

        lr = LR
        for pg in opt.param_groups: pg['lr'] = lr

        x, y = train_data.get_batch()
        opt.zero_grad()
        if USE_AUTOCAST:
            with torch.autocast(device_type='cuda', dtype=DTYPE):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        else:
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
            last_d_eff = measure_d_eff(model)
            model.train()
            logger.log(run, global_step+1, model.d, last_d_eff, ce, val, lr,
                       model.param_count, flops_used)

            half = eval_interval_count
            progress = None
            if not in_recovery and len(val_history) >= 2 * half:
                old_window = val_history[-2*half:-half]
                new_window = val_history[-half:]
                best_old = min(v for _, v in old_window)
                best_new = min(v for _, v in new_window)
                progress = best_old - best_new

            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_used = torch.cuda.memory_allocated()
            vram_ratio = vram_used / vram_total
            vram_capped = vram_ratio > VRAM_CAP_RATIO

            in_ramp = model.d_active < model.d_active_target
            recov_tag = ' [recov]' if in_recovery else ''
            ramp_tag = f' [ramp d_a={model.d_active:.2f}/{model.d_active_target}]' if in_ramp else ''
            cap_tag = ' [VRAM-CAP]' if vram_capped else ''
            div_tag = ' [DIV]' if (val - best) > 2 * SIGMA_EVAL else ''
            prog_str = f'prog={progress:+.3f}' if progress is not None else 'prog=NA'
            print(f'  >>> eval {global_step+1} d={model.d} val={val:.4f} '
                  f'best={best:.4f} {prog_str} d_eff_W={last_d_eff:.3f}'
                  f'{recov_tag}{ramp_tag}{cap_tag}{div_tag} '
                  f'vram={vram_ratio*100:.0f}% '
                  f'FLOP={flops_used/TARGET_FLOPS*100:.1f}%', flush=True)

            # Trigger: stalled AND ramp complete (no grow mid-ramp).
            stalled = progress is not None and progress < PROGRESS_THRESHOLD
            if (stalled and not in_ramp and model.d < D_MAX
                    and not vram_capped):
                target = math.ceil(model.d * GROWTH_FACTOR / 2) * 2
                new_d = min(max(model.d + GROWTH_MIN_STEP, target), D_MAX)
                new_d = round_d_to_valid(new_d)
                print(f'  *** GROW d={model.d}→{new_d} '
                      f'(prog={progress:+.3f}<{PROGRESS_THRESHOLD}; '
                      f'd_eff={last_d_eff:.3f}; '
                      f'val-best={val-best:+.3f}; '
                      f'ramp K_per_dim={K_PER_DIM}) ***',
                      flush=True)
                model.grow(new_d)
                model = model.to(DEVICE)
                opt = make_opt(model)
                last_growth_step = global_step
                model.last_ramp_step = global_step
                val_history = []
                print(f'  params={model.param_count/1e6:.2f}M, '
                      f'd_active={model.d_active:.2f}→{model.d_active_target} '
                      f'over {int((new_d - model.d_active)*K_PER_DIM)} steps',
                      flush=True)

        if (global_step + 1) % 5000 == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            path = os.path.join(CKPT_DIR, f'{CKPT_PREFIX}step{global_step+1}.pt')
            torch.save({'model': model.state_dict(), 'step': global_step+1,
                        'd': model.d, 'flops_used': flops_used, 'best': best,
                        'wall': time.time() - t0},
                       path)
            print(f'  [ckpt] saved {path}', flush=True)

        global_step += 1

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
    print(f'{"="*60}')

    train_data.close(); eval_data.close()

if __name__ == '__main__':
    main()
