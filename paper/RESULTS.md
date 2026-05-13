# Adaptive Width Growth — Results

Single 22-layer adaptive run on FineWeb-Edu (2.9B tokens, GPT-2 BPE,
context 512), grown from `d=2` under the math-derived trigger of the
paper.  Hardware: one RTX PRO 6000 Blackwell, 102 GB.

## Headline

Best at matched wall-clock 6.81 h:

| run | best val CE | FLOPs to best | wall to best |
|---|---:|---:|---:|
| Adaptive start `d=2`, grow to `d=496` | **3.565** | 1.00×10¹⁸ | 6.81 h |
| Adaptive start `d=64`, grow to `d=496` | 3.628 | 1.20×10¹⁸ | 6.51 h |
| Fixed `d=544` E2E, no growth | 3.855 | 1.66×10¹⁸ | 5.92 h |

The adaptive run starting at `d=2` wins both FLOPs-to-best and
wall-clock-to-best; see [`adapt_vs_e2e.pdf`](adapt_vs_e2e.pdf) and
[`startd_ablation.pdf`](startd_ablation.pdf).

## Best per `d` along the adaptive trajectory

Each row is the best val CE the adaptive run achieved while at that
width, with the FLOPs spent up to that point.  FLOPs are absolute
(in $10^{16}$ units).

| d | best val CE | FLOPs ($10^{16}$) |
|---:|---:|---:|
|   2 | 7.71 | 0.01 |
|   8 | 6.91 | 0.07 |
|  16 | 6.43 | 0.29 |
|  32 | 5.92 | 1.1  |
|  64 | 5.34 | 1.6  |
| 128 | 4.50 | 5.0  |
| 256 | 3.97 | 30   |
| 336 | 3.75 | 50   |
| **496** | **3.565** | **100** |
| 544 | 3.63 | 122  |

Past `d ≈ 544` the trajectory plateaus near the Chinchilla floor
predicted for our $(N, D)$ pair ($\approx 3.46$ at $N \approx 105$M,
$D = 2.9$B); further val descent requires more training tokens, not
more model size.  The divergence guard in the trigger holds off
growth once the model starts drifting above its running best, so the
trajectory plateaus rather than blows up.

## On comparing against a fixed-`d=2048` baseline

An earlier framing of this work reported FLOP-efficiency relative to
a fixed-`d=2048`, 22-layer E2E baseline run for 25 000 steps (val
4.97 at step 24 000).  That baseline is data-bottlenecked at 2.9 B
tokens ($\approx 0.7$ tokens/param, vs. Chinchilla-optimal $\approx
20$), so it is undertrained capacity rather than a tight ceiling.
We do not lead with that comparison in the paper.  The same-`d=544`
baseline above (matched wall-clock, same training stack) is the
honest comparison.

## Reproduce

```
python adaptive_growth.py
```

Override `CACHE_PATH`, `DB_PATH`, `CKPT_DIR`, `RUN_NAME` via
environment variables.  See the paper for the trigger constants,
LR-scaling argument, asymmetric grow init, precision diagnostic, and
stepped `d_head` schedule.
