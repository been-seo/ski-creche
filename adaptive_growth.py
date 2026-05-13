"""
Adaptive progressive width growth for Snowman language models, with a
math-derived trigger.  See paper/paper.tex for the full derivation.

  - Trigger: best progress in window W < k·σ_eval → grow.
  - Growth:  d_new = round_to_next_d_head(max(d+2, ceil(d·α/2)*2)).
  - Recovery: W steps after grow with no trigger check.
  - Init:    asymmetric — noise on new rows of W_{q,k,v} and ff1;
             zero on new output rows of W_o and ff2 (the gates).
  - Stepped d_head(d) so n_heads stays bounded at 16.

Environment variables (all optional):
  CACHE_PATH, DB_PATH, CKPT_DIR, RUN_NAME, START_D.
"""
import math, os, queue, sqlite3, threading, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = 'cuda'
DTYPE = torch.bfloat16
V = 50257
D_HEAD = 2  # minimum d_head; actual d_head grows with d via d_head_for(d)
N_LAYERS = 22

def d_head_for(d):
    """d_head ramps up with d to keep n_heads bounded.
    n_heads = d / d_head; for our seq=512 attention memory is O(B·n_heads·T²),
    so we cap it via stepped d_head."""
    if d <= 8:    return 2   # heads up to 4
    if d <= 32:   return 4   # heads up to 8
    if d <= 128:  return 8   # heads up to 16
    if d <= 512:  return 16  # heads up to 32
    return 32                 # heads up to 64 at d=2048

def round_d_to_valid(target_d):
    """Round target_d up to nearest multiple of d_head_for(target_d)."""
    dh = d_head_for(target_d)
    return ((target_d + dh - 1) // dh) * dh
BATCH, SEQ_LEN = 64, 512
LR, WD, GRAD_CLIP = 1.5e-4, 0.01, 1.0
USE_8BIT = False  # fp32 AdamW — bf16+8bit causes systematic drift per H4
USE_AUTOCAST = True  # bf16 forward (saves VRAM), fp32 AdamW state
D_MAX = 2048
GROWTH_FACTOR = 1.05  # multiplicative growth (5% more dim per grow)
GROWTH_MIN_STEP = 2   # absolute floor for small d
WINDOW_STEPS = 2000   # progress measurement window
SIGMA_EVAL = 0.05     # eval-batch noise floor (std of val CE across batches)
NOISE_K = 2.0         # threshold = NOISE_K · SIGMA_EVAL
PROGRESS_THRESHOLD = NOISE_K * SIGMA_EVAL  # 0.10 default
DEFF_THRESHOLD = 0.30  # weight-rank ratio: grow only if d_eff_W/d > this
                       # (trained transformer weights settle around 0.3-0.5;
                       # lower threshold keeps trigger active across regime)
ROW_NOISE = 1e-3  # asymmetric init: new rows of W_q/W_k/W_v/ff1 get this·σ_w noise
VRAM_CAP_RATIO = 0.90  # stop growing when VRAM > this fraction

CACHE_PATH = os.environ.get('CACHE_PATH', 'data_cache/fineweb_2.9B.pt')
EVAL_TOKENS = 1_000_000
DB_PATH = os.environ.get('DB_PATH', 'adaptive.db')
CKPT_PREFIX = 'adaptive_'
CKPT_DIR = os.environ.get('CKPT_DIR', 'ckpts')
EVAL_INTERVAL = 500
EVAL_BATCHES = 20
LOG_INTERVAL = 100

# FLOP reference for the per-step FLOP% log line (display only — not used
# in any decision).  Default: a fictitious d=2048, 22L, 25 000-step run;
# override with TARGET_FLOPS env var if you want a different reference.
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
    """LayerNorm restricted to active dimensions (PAVING width-elasticity proof).
    Stats (mean/var) are computed over first d_active dims only.
    Combined with zero-pad weights → growth is function-preserving."""
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.d_active = d
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        d = self.d_active
        active = x[..., :d]
        mu = active.mean(dim=-1, keepdim=True)
        var = active.var(dim=-1, keepdim=True, unbiased=False)
        normed = (active - mu) / torch.sqrt(var + self.eps)
        out = torch.zeros_like(x)
        out[..., :d] = self.weight[:d] * normed + self.bias[:d]
        return out

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

def _grow_linear(layer, new_in, new_out, noise_new_rows=0.0, clone_new_rows=False):
    """Grow a Linear from (old_out, old_in) to (new_out, new_in).

    Existing block (rows < old_out, cols < old_in) is copied verbatim.
    Behaviour for the new rows (rows in [old_out, new_out)) is controlled
    by `clone_new_rows` and `noise_new_rows`:

      clone_new_rows=False (zero-pad + optional small noise):
        New rows are zero except for noise_new_rows * sigma_w * N(0,1) on
        the first old_in cols.  Used when the layer's output is gated
        (W_o, ff2) and we want pure zero — set noise=0.  Used with noise
        when the layer's output drives attention (W_q,W_k,W_v) and we
        need a tiny signal to wake the gate up.

      clone_new_rows=True (Net2Net):
        For each new row i, pick a random source row s_i ~ U[0, old_out)
        and copy that row's first old_in cols verbatim, plus
        noise_new_rows * sigma_w * N(0,1) symmetry-breaking perturbation.
        New heads start as duplicates of existing heads — they immediately
        produce a meaningful attention pattern instead of random noise,
        so the gradient signal at the gate (W_o new col) is the
        gradient that the source head already receives.  After a few
        steps the gradient noise breaks the symmetry.

    Cols beyond old_in on existing rows are always zero (so the layer
    does not read new input dimensions; that happens on the consumer
    side by the next layer's grow).
    """
    old_w = layer.weight.data
    old_out, old_in = old_w.shape
    dev = old_w.device
    w = torch.zeros(new_out, new_in, device=dev)
    w[:old_out, :old_in] = old_w
    n_new = new_out - old_out
    if n_new > 0:
        if clone_new_rows:
            src = torch.randint(0, old_out, (n_new,), device=dev)
            w[old_out:, :old_in] = old_w[src, :]
        if noise_new_rows > 0:
            scale = noise_new_rows
            if old_w.numel() > 1:
                scale = noise_new_rows * old_w.std().item()
            w[old_out:, :old_in] += torch.randn(n_new, old_in, device=dev) * scale
    layer.weight = nn.Parameter(w)
    if layer.bias is not None:
        b = torch.zeros(new_out, device=dev)
        b[:old_out] = layer.bias.data
        if clone_new_rows and n_new > 0:
            src = torch.randint(0, old_out, (n_new,), device=dev)
            b[old_out:] = layer.bias.data[src]
        layer.bias = nn.Parameter(b)
    layer.in_features, layer.out_features = new_in, new_out

def _grow_active_ln(ln, new_d, activate=True):
    """ActiveLayerNorm grow: extend gain/bias to new_d (gain=1, bias=0 for new).
    activate=True: set d_active=new_d (one-time mean/var shock from including zero
    new dims in stats). activate=False: keep d_active at old (function preserved
    but new dims dead — for two-phase protocol)."""
    old_d = ln.weight.shape[0]
    dev = ln.weight.device
    w = torch.ones(new_d, device=dev)
    b = torch.zeros(new_d, device=dev)
    w[:old_d] = ln.weight.data; b[:old_d] = ln.bias.data
    ln.weight = nn.Parameter(w); ln.bias = nn.Parameter(b)
    if activate:
        ln.d_active = new_d

def grow_block(block, new_d):
    """Asymmetric grow: noise on NEW ROWS of {W_q,W_k,W_v, ff1}, all else zero.
    W_o new cols stay 0 (gate), ff2 new rows/cols stay 0 — preserves function
    while letting the gates start receiving gradient."""
    old_d = block.d
    _grow_active_ln(block.ln1, new_d)
    _grow_active_ln(block.ln2, new_d)
    # q/k/v: small noise on new rows (creates signal in new heads)
    _grow_linear(block.W_q, new_d, new_d, noise_new_rows=ROW_NOISE, clone_new_rows=True)
    _grow_linear(block.W_k, new_d, new_d, noise_new_rows=ROW_NOISE, clone_new_rows=True)
    _grow_linear(block.W_v, new_d, new_d, noise_new_rows=ROW_NOISE, clone_new_rows=True)
    # W_o: gate — keep all-zero new entries
    _grow_linear(block.W_o, new_d, new_d, noise_new_rows=0.0)
    # ff1 new rows = noise (creates signal in new intermediate dims)
    _grow_linear(block.ff1, new_d, 4*new_d, noise_new_rows=ROW_NOISE, clone_new_rows=True)
    # ff2: gate — keep new rows AND new cols at zero
    _grow_linear(block.ff2, 4*new_d, new_d, noise_new_rows=0.0)
    block.d = new_d
    block.d_head = d_head_for(new_d)
    block.n_heads = new_d // block.d_head

class Model(nn.Module):
    def __init__(self, d, n_layers, seq_len):
        super().__init__()
        self.d = d
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
        """Forward pass returning final hidden states for d_eff measurement."""
        h = self.embed(idx) + self.pe[:,:idx.size(1)]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h)), h
    def grow(self, new_d):
        old_d = self.d; dev = self.embed.weight.device
        # Embed: zero-pad new cols (function-preserving with ActiveLN)
        old_emb = self.embed.weight.data
        w = torch.zeros(V, new_d, device=dev)
        w[:, :old_d] = old_emb
        self.embed = nn.Embedding(V, new_d).to(dev)
        self.embed.weight = nn.Parameter(w)
        # PE: zero-pad new dims (so x_new = 0 → ActiveLN unaffected)
        new_pe = torch.zeros(1, SEQ_LEN, new_d, device=dev)
        new_pe[:,:,:old_d] = self.pe[:,:SEQ_LEN,:old_d]
        self.pe = new_pe
        for block in self.blocks:
            grow_block(block, new_d)
        _grow_active_ln(self.ln_f, new_d)
        # head shares embed weight; just update Linear metadata
        self.head = nn.Linear(new_d, V, bias=False).to(dev)
        self.head.weight = self.embed.weight
        self.d = new_d
    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

@torch.no_grad()
@torch.no_grad()
def measure_d_eff(model, data=None):
    """Effective rank of WEIGHT matrices (PE-clean, batch-free, near-zero cost).
    Aggregates entropy-based effective rank across W_q/k/v/o, ff1, ff2 of every
    block, normalized by min(W.shape) so ratio is in [0, 1]. Returns (mean,
    last_block_ratios) — the absolute mean is the trigger signal."""
    ratios = []
    for block in model.blocks:
        for W in (block.W_q.weight, block.W_k.weight,
                  block.W_v.weight, block.W_o.weight,
                  block.ff1.weight, block.ff2.weight):
            s = torch.linalg.svdvals(W.float())
            s_sum = s.sum()
            if s_sum < 1e-10:
                ratios.append(0.0)
                continue
            p = s / s_sum
            p = p[p > 1e-10]
            entropy = -(p * p.log()).sum()
            d_eff = entropy.exp().item()
            ratios.append(d_eff / min(W.shape))
    return sum(ratios) / max(len(ratios), 1)

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
        # wall offset: prefer (a) value saved in ckpt, else (b) derive from
        # ckpt-file mtime vs DB first-ts, else (c) WALL_OFFSET env var, else 0
        wall_offset = ckpt.get('wall')
        if wall_offset is None:
            try:
                conn = sqlite3.connect(DB_PATH)
                first_ts = conn.execute('SELECT MIN(ts) FROM log').fetchone()[0]
                conn.close()
                if first_ts is not None:
                    wall_offset = os.path.getmtime(args.resume) - first_ts
            except Exception:
                wall_offset = None
        wall_offset = float(os.environ.get('WALL_OFFSET', wall_offset or 0.0))
        print(f'  Resumed: step={global_step}, d={ckpt["d"]}, flops={flops_used:.2e}, wall_offset={wall_offset/3600:.2f}h')
    else:
        start_d = int(os.environ.get('START_D', 2))
        model = Model(start_d, N_LAYERS, SEQ_LEN).to(DEVICE)
        global_step = 0
        flops_used = 0
        best = float('inf')

    print(f'Model: d={model.d}, {N_LAYERS}L, {model.param_count/1e6:.2f}M params')

    # LR scales as eta(d) = LR * sqrt(d0 / d) per the |ΔL| ≤ η sqrt(N)
    # scaling argument (§3.5).  d0 = start width.
    LR_D0 = max(1, model.d)  # reference width
    import math as _math
    def lr_for(d):
        return LR * _math.sqrt(LR_D0 / d)

    if USE_8BIT:
        import bitsandbytes as bnb
        make_opt = lambda m, lr: bnb.optim.AdamW8bit(m.parameters(), lr=lr, weight_decay=WD)
    else:
        make_opt = lambda m, lr: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)

    opt = make_opt(model, lr_for(model.d))
    t0 = time.time() - wall_offset
    run = os.environ.get('RUN_NAME', 'adaptive')
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
        # Using fp32 (no precision drift) so we can use the conventional
        # LR=3e-4 directly. The sqrt(N) bound is conservative and not
        # binding here per the measurement of gradient effective rank.
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

        # Weight-rank d_eff: PE-clean, batch-free, used in hybrid grow trigger.
        # Measured at every eval; cheap (sum of SVDs, ~ms even at d=2048).

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
            # Weight-rank d_eff (always measured at eval — used for hybrid trigger)
            last_d_eff = measure_d_eff(model)
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
            div_tag = ' [DIV]' if (val - best) > 2 * SIGMA_EVAL else ''
            prog_str = f'prog={progress:+.3f}' if progress is not None else 'prog=NA'
            print(f'  >>> eval {global_step+1} d={model.d} val={val:.4f} '
                  f'best={best:.4f} {prog_str} d_eff_W={last_d_eff:.3f}'
                  f'{recov_tag}{cap_tag}{div_tag} '
                  f'vram={vram_ratio*100:.0f}% '
                  f'FLOP={flops_used/TARGET_FLOPS*100:.1f}%', flush=True)

            # Hybrid trigger: progress plateau AND weights filling capacity AND
            # not currently diverging from best.
            # Divergence guard prevents firing on a degrading val curve where
            # best is stuck but recent vals are getting worse — growing then
            # only compounds the damage.
            stalled = progress is not None and progress < PROGRESS_THRESHOLD
            saturated = last_d_eff > DEFF_THRESHOLD
            diverging = (val - best) > 2 * SIGMA_EVAL  # 0.10 above best
            if (stalled and saturated and model.d < D_MAX
                    and not vram_capped and not diverging):
                # multiplicative growth, rounded up to be divisible by the
                # next d_head (so n_heads is integer)
                target = math.ceil(model.d * GROWTH_FACTOR / 2) * 2
                new_d = min(max(model.d + GROWTH_MIN_STEP, target), D_MAX)
                new_d = round_d_to_valid(new_d)
                print(f'  *** GROW d={model.d}→{new_d} '
                      f'(prog={progress:+.3f}<{PROGRESS_THRESHOLD} '
                      f'∧ d_eff={last_d_eff:.3f}>{DEFF_THRESHOLD} '
                      f'∧ val-best={val-best:+.3f}) ***',
                      flush=True)
                model.grow(new_d)
                model = model.to(DEVICE)
                opt = make_opt(model, lr_for(model.d))
                print(f'  lr={lr_for(model.d):.2e} (scaled by sqrt({LR_D0}/{model.d}))', flush=True)
                last_growth_step = global_step
                val_history = []  # reset history at new d
                print(f'  params={model.param_count/1e6:.2f}M', flush=True)

        # checkpoint
        if (global_step + 1) % 5000 == 0:
            os.makedirs(CKPT_DIR, exist_ok=True)
            path = os.path.join(CKPT_DIR, f'{CKPT_PREFIX}step{global_step+1}.pt')
            torch.save({'model': model.state_dict(), 'step': global_step+1,
                        'd': model.d, 'flops_used': flops_used, 'best': best,
                        'wall': time.time() - t0},
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
