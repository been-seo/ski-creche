# Ski Crèche

Based on [doi:10.5281/zenodo.19454889](https://doi.org/10.5281/zenodo.19454889)

---

Train a deep network without holding deep activations in memory.

A single block is weight-tied and applied K times. Each depth is trained
independently: detach the input, forward one block, compute loss, backprop,
free. Only one depth's activations live on GPU at a time — O(1) VRAM
regardless of K. Depth and width are elastic: both can be grown or shrunk
without retraining.

## Why it works

Three properties make this non-trivial:

**Single-Depth Sufficiency** (corollary of Theorem 13,
[doi:10.5281/zenodo.19454889](https://doi.org/10.5281/zenodo.19454889)).
Training at a randomly sampled depth `k ~ U{1..K}` gives the same update
*direction* as training all depths simultaneously. Single-depth sampling
scales `gᵢ` by `1/K`; since `vᵢ = EMA(gᵢ²)` scales by `1/K²`, the ratio
`mᵢ/√vᵢ` — the AdamW update direction — is invariant to `K`. Theorem 13
establishes `vᵢ ≈ σᵢ²` (Jacobian column norm), making AdamW a diagonal
Gauss–Newton method; Remark 15 identifies this as σ-normalised gradient
descent. The scale cancellation is exact, not approximate. *Does not hold
for SGD.*

**No spurious local minima** (Theorem 35 + Theorem 41 [Singular Escape
Lemma], ibid.). Treating K depths as K tasks: at any regular critical point
(rank(J) = K), L = 0 by Theorem 35. For d > 2K, any singular critical point
with L ≠ 0 is not a local minimum — Theorem 41 guarantees an escape
direction of codimension ≤ d − 2K. Together they eliminate all spurious
local minima for d > 2K.

**Depth Detach is Free.** Per-depth backward with `h.detach()` between
depths makes each depth independent. Depth `k` does not depend on depths
`k+1..K` being present. After training, the model at depth 2 is *exactly*
the model that was trained at depth 2. No surgery, no retraining — just
call `forward(x, K=2)`.

**Width Detach is Free** (with width-sampled training). At width `g`,
gradients flow only through `W[:d_g, :d_g]`. After training with
`g ~ U{1..G}`, the submatrix `W[:d_g, :d_g]` has been optimised by
gradients from all widths `g' ≥ g`. Width detach = restrict to a subspace
that was trained. Proof in [`proofs/width_detach.py`](proofs/width_detach.py).

These three properties are symmetric:

| Property      | Depth                        | Width                        |
|---------------|------------------------------|------------------------------|
| Shared params | Block W (all depths)         | W[:d_g,:d_g] (all g' ≥ g)   |
| Isolation     | `h.detach()`                 | Width restriction + ALN      |
| Sampling      | k ~ U{1..K}                  | g ~ U{1..G}                  |
| Detach cost   | Free                         | Free                         |

---

## Snowman

`Snowman` is the model. It holds three user-provided components and a set
of learnable gates:

```
h₀ = Embed(x)
h_k = h_{k-1} + α_k · (Block(h_{k-1}) - h_{k-1})    for k = 1..K
logits = Readout(h_K)
```

- **Embed**: maps input tokens to hidden states. `nn.Module`, `forward(x) -> h`.
  Use fixed sinusoidal position encoding — learnable pos embeddings are dead
  in `forward_single` because `h.detach()` severs the gradient path before
  depth 0.
- **Block**: the repeated unit. Any `nn.Module` mapping `h -> h'` with the
  same shape. Transformer, conv, MLP, SSM — the library doesn't care.
- **Readout**: maps hidden state to logits. Must tie weights with Embed
  (see FLOP cost below).
- **Gates** α_k: learnable scalars. α_0 = 1 (block active from the start),
  α_{k>0} = 0 (identity until grown into). When α_k = 0, h_k = h_{k-1}
  exactly — adding depth can never increase loss.

`forward(x, K)` runs the full stack for inference. `forward_single(x, k,
targets, loss_fn)` trains at a single depth: intermediate depths run under
`torch.no_grad()`, the final depth runs with full gradient tracking, then
`.backward()` is called immediately and activations are freed.

---

## Snowball schedule

`SnowballTrainer` manages growth. Training has K_max phases:

1. **Phase 1**: K=1, train for S steps with cosine LR decay.
2. **Phase 2**: K=2, reset optimizer, new cosine schedule. Gate α_1 = 0.
3. ...
4. **Phase K_max**: final phase.

Within each phase K, depth `k` is sampled uniformly from `{1..K}` each step.
Every depth receives gradients in expectation; no depth is neglected as K
grows.

S (steps per phase) is set directly (`total_steps`) or derived from a FLOP
budget: `S = budget / (tokens_per_step × Σ_{k=1}^{K_max} flop_single(k))` —
equal steps per phase, so larger K values get sufficient training.

---

## FLOP cost

Weight tying is not optional. Without it the readout head costs V×d at every
depth, making local learning more expensive than E2E.

With tied weights and single-depth sampling:

```
single(K) = 2·P_emb + (2K+2)·B + 4·P_head
e2e(L)    = 6·(P_emb + L·B + P_head)
```

- `2·P_emb`: embed forward only (detached, no backward).
- `(2K+2)·B`: K−1 block forwards without grad + 1 forward+backward.
- `4·P_head`: one readout forward+backward. P_head includes the tied V×d
  matmul.

Averaged over k=1..K_max, single-depth sampling costs ~0.54× the E2E FLOP
at matching depth.

---

## VRAM

Per-depth backward holds only one block application's activations at a time.
E2E holds O(K) activations; Snowball holds O(1).

Measured on RTX 5070 Ti, d=384, seq=256, batch=32, transformer block:

| K | E2E      | Snowball | ratio |
|---|----------|----------|-------|
| 2 | 7,294 MB | 5,591 MB | 0.77× |
| 4 | 7,896 MB | 5,591 MB | 0.71× |
| 8 | 9,034 MB | 5,591 MB | 0.62× |

Snowball peak is constant across K.

---

## Interfaces

```python
from ski_creche import Block, DataStream

class MyBlock(Block):           # h -> h', same shape
    def forward(self, h): ...

class MyData(DataStream):       # -> (x, y) tensors
    def get_batch(self): ...
```

Embed and Readout are plain `nn.Module` — no ABC needed.
Readout must reuse Embed's weight matrix.

---

## Usage

```python
from ski_creche import Block, DataStream, Snowman, SnowballTrainer, SnowballConfig

embed = MyEmbed(vocab, d)
block = MyBlock(d)
readout = MyReadout(d, embed.weight)  # tie weights
model = Snowman(embed, block, readout, k_max=8)

cfg = SnowballConfig(
    k_max=8, lr=3e-4, weight_decay=0.01, grad_clip=1.0,
    eval_interval=200, log_interval=200, eval_batches=10,
    db_path='run.db',
    checkpoint_dir='ckpts/',
    checkpoint_interval=1000,
)
trainer = SnowballTrainer(model, cfg, train_data, eval_data,
                          tokens_per_step=batch * seq_len)
result = trainer.train(flop_budget=budget)
```

All config fields default to `None`. `validate()` checks required fields
before training starts.

### Config

| Field                | Default       | What it does                                    |
|----------------------|---------------|-------------------------------------------------|
| `k_max`              | None          | Maximum depth                                   |
| `lr`                 | None          | Peak learning rate (cosine-decayed per phase)   |
| `weight_decay`       | None          | AdamW weight decay                              |
| `grad_clip`          | None          | Max gradient norm                               |
| `start_k`            | 1             | Start from this depth (skip earlier phases)     |
| `total_steps`        | None          | Steps per phase (alternative to FLOP budget)    |
| `eval_interval`      | None          | Evaluate every N steps                          |
| `log_interval`       | None          | Log every N steps                               |
| `eval_batches`       | None          | Number of batches per evaluation                |
| `loss_fn`            | cross-entropy | Any `(logits, targets) -> scalar`               |
| `optimizer_factory`  | AdamW         | Custom optimizer constructor                    |
| `db_path`            | None          | SQLite path for logging                         |
| `checkpoint_dir`     | None          | Directory for checkpoint `.pt` files            |
| `checkpoint_interval`| None          | Save every N steps                              |
| `checkpoint_on_phase`| True          | Save on phase transition                        |
| `checkpoint_best`    | True          | Save on best eval loss                          |

### Callbacks

```python
trainer = SnowballTrainer(model, cfg, train_data, eval_data,
    on_step=lambda step, K, ce, lr, flops: ...,
    on_eval=lambda step, K, val_ce, flops: ...,
    on_phase=lambda old_K, new_K, step, flops: ...,
)
```

---

## Elastic depth

**Attach** — load a K=4 checkpoint and resume training from K=5:

```python
model = Snowman(embed, block, readout, k_max=4)
model.load_state_dict(torch.load('ckpt_K4.pt'))
model.resize_depth(8)          # gates 4..7 init to 0 (identity)

cfg = SnowballConfig(k_max=8, start_k=5, ...)
trainer = SnowballTrainer(model, cfg, ...)
trainer.train()
```

**Detach** — run at smaller depth with no retraining:

```python
model.load_state_dict(torch.load('ckpt_K8.pt'))
logits = model(x, K=2)        # upper gates ignored
```

This is free because `forward_single` trains each depth on its own loss.
Depth k does not depend on depths k+1..K existing.

FineWeb-Edu, d=384, 500 steps/phase:

|                        | best CE |
|------------------------|---------|
| Base K=4               | 7.67    |
| Attach K=4→8           | 7.48    |

Depth detach (K=4 checkpoint, no retraining):

| K | CE   |
|---|------|
| 1 | 7.90 |
| 2 | 8.06 |
| 3 | 7.85 |
| 4 | 7.95 |

See [`examples/elastic_depth.py`](examples/elastic_depth.py).

---

## Elastic width

Width (hidden dimension d) is also elastic. The widened model produces
output *identical* to the original for all inputs — proven in
[`proofs/width_elasticity.py`](proofs/width_elasticity.py).

**Theorem (Exact Width Preservation).** Let M_d be a Snowman with hidden
dimension d and n_h attention heads (d_head = d/n_h). Define M_{d'} with
d' = d + n_new·d_head by zero-padding embedding and FF weights, adding n_new
zero-init attention heads, and replacing LayerNorm with ActiveLayerNorm
(normalises over the first d dims only). Then M_{d'}(x, K) = M_d(x, K) for
all inputs and depths. (Proof: by induction on layers. ActiveLayerNorm is
essential — standard LayerNorm breaks preservation because zero-padded
dimensions corrupt the normalisation statistics.)

Width must grow by adding new heads, not widening existing ones. Widening
changes d_head and thus the attention scaling 1/√d_head, breaking exact
preservation. This constrains d to grow in multiples of d_head.

**Attach** — grow d while preserving all learned representations:

```python
model.grow_width(d_new=384)   # zero-pads all layers, ALN activates at old d
model.activate_width()        # ALN normalises over new d
```

**Detach** — run at smaller width. With width-sampled training, no retraining
needed (proven in [`proofs/width_detach.py`](proofs/width_detach.py)).

FineWeb-Edu, K=8, seq=256, batch=16, 4800 total steps:

|              | Progressive (d: 192→384→768) | Baseline (d=768) |
|--------------|------------------------------|------------------|
| Best CE      | **7.27**                     | 7.44             |
| Params       | 45.7M                        | 45.7M            |
| Time         | **287s**                     | 372s             |

Progressive starts at d=192 (1600 steps), grows to d=384, then d=768 (1600
steps each). Better CE and 23% faster — early steps use smaller matrices.

VRAM per width phase:

| d   | VRAM     |
|-----|----------|
| 192 | 3,563 MB |
| 384 | 3,772 MB |
| 768 | 4,219 MB |

Training can start on a smaller GPU and move up as width grows.

See [`examples/fineweb_elastic.py`](examples/fineweb_elastic.py).

---

## FineWeb-Edu results

```bash
python examples/fineweb.py
```

d=384, 6 heads, K=8, seq=256, batch=32. FLOP-matched against an 8-layer
untied transformer:

|           | E2E     | Snowball |
|-----------|---------|----------|
| Best CE   | 7.42    | 7.33     |
| Params    | 33.5M   | 21.1M    |
| FLOP      | 5.2e15  | 4.9e15   |
| Peak VRAM | 9,303 MB| 5,592 MB |

Convergence per phase: 13.9 → 9.2 → 8.3 → 8.0 → 7.4 → 7.4 → 7.4 → 7.3

d=768, 12 heads, K=8, seq=256, batch=16 (GPT-2 Small scale):

|           | E2E     | Snowball |
|-----------|---------|----------|
| Best CE   | 7.71    | 7.43     |
| Params    | 95.3M   | 45.7M    |
| FLOP      | 6.6e15  | 6.0e15   |
| Peak VRAM | 6,766 MB| 3,420 MB |

Convergence per phase: 12.7 → 9.4 → 8.5 → 8.1 → 7.8 → 7.4 → 7.5 → 7.7

Snowball advantage grows with scale: CE gap widens (−0.09 → −0.28), VRAM
ratio improves (0.62× → 0.51×), parameter ratio improves (0.63× → 0.48×).

---

## Monitoring

```bash
python -m ski_creche run.db              # latest run status
python -m ski_creche run.db --compare    # compare all runs in db
python -m ski_creche run.db --run NAME   # specific run
```

---

## Adaptive scale-up

PAVING-style progressive width growth on FineWeb-Edu, with an
automatic math-derived growth scheduler so the user never picks a
`d` schedule by hand.  The model starts at `d=2` and grows on demand:
each grow event fires when the per-FLOP information gain over a
sliding window drops below the eval-batch noise floor.

### Setup

| | |
|---|---|
| Architecture | 22-layer transformer, sinusoidal PE, weight-tied head |
| Optimizer | fp32 AdamW, `lr=1.5e-4`, `wd=0.01`, `grad_clip=1.0` |
| `d_head` | stepped: 2 (d≤8), 4 (d≤32), 8 (d≤128), 16 (d≤512), 32 (d>512) |
| Data | FineWeb-Edu sample-10BT, GPT-2 BPE, 2.9B tokens cached |
| Eval | 20-batch CE on held-out 1M tokens, σ_eval ≈ 0.05 |
| Hardware | 1× NVIDIA RTX PRO 6000 Blackwell, 102 GB VRAM |

### Trigger (no hand-tuned thresholds)

```
W = 2000 steps  (≈ 4 evals at EVAL_INTERVAL=500)
threshold = k · σ_eval = 2 · 0.05 = 0.10

best_old   = min(val_ce in [t-2W, t-W])
best_new   = min(val_ce in [t-W, t])
progress   = best_old - best_new
d_eff_W    = entropy-based effective rank of block weights
div_guard  = (val_t - best) < 2·σ_eval

if progress < threshold and d_eff_W > 0.3 and div_guard:
    new_d = round_to_d_head_multiple( max(d+2, ⌈d · 1.05 / 2⌉ · 2) )
    grow()  # noise init on W_{q,k,v} new rows, zero on W_o/ff2 output gates
```

### Result vs same-`d` E2E baseline

We train a fixed-width `d=544` end-to-end baseline from scratch with
identical data stream, tokenizer, context length, optimizer, learning
rate, grad clip, and hardware, and run for matched wall-clock 6.81 h.
`d=544` matches the final width the adaptive run grew into.

Both runs trained for the same total wall-clock 6.81 h on the same GPU.

| | best val CE | FLOPs to best | wall to best |
|---|---:|---:|---:|
| Adaptive (d=2→496) | **3.565** | 1.00×10¹⁸ | 6.81 h |
| Fixed d=544 E2E   | 3.855      | 1.66×10¹⁸ | 5.92 h |

(Adaptive's best is its last eval — still descending when stopped.
E2E's best comes earlier and does not improve over the remaining
~1 h of wall, so the 6.81 h vs 5.92 h gap reflects E2E plateauing,
not less compute spent on it.)

The adaptive run wins by **−0.29 val CE** while also using **~1.7×
fewer FLOPs** to reach its best.  Past d=544 the trajectory plateaus
near the Chinchilla floor predicted for our (N, D) pair (≈ 3.46),
so further val descent requires more training tokens, not more model
size.

See [`scaleup/paper.pdf`](scaleup/paper.pdf) for the LR-bound
derivation `η_max = σ_eval / √N`, the asymmetric grow init that
breaks the dead-attention-head failure mode of zero-pad, the
precision diagnostic (`bf16+AdamW8bit` introduces systematic
optimizer-state drift; `fp32 AdamW` removes it), and the figures.

---

## Rust

Same algorithm, same DB schema. Uses `burn` (pure Rust, wgpu backend) and
`rusqlite`. No C++ dependencies — runs on any GPU via WebGPU/Vulkan.

```bash
cd rust
cargo run --example basic --release
```

```rust
use ski_creche::model::SnowmanConfig;

let model = SnowmanConfig::new(vocab, d, d_ff, seq_len, k_max)
    .init::<B>(&device);
let logits = model.forward(x, k);
```

Examples: `basic` (K=1→4), `vram_compare` (O(1) VRAM demo),
`elastic_depth` (attach/detach), `elastic_width` (width growth).

---

## License

MIT
