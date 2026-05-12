# Scale-up Run Results

Single 22-layer adaptive run on FineWeb-Edu, FLOP-matched to a 5.95 PFLOPs
E2E baseline (d=2048, 22L, 25 000 steps, val 4.965 at step 24 000,
val 5.18 at step 25 000 — see Appendix on E2E non-convergence).

## Best per `d` (adaptive run)

| d | best val CE | FLOP used | FLOP % |
|---:|---:|---:|---:|
| 2 | 7.753 | 1.0e+13 | 0.0002 |
| 4 | 7.605 | 1.8e+14 | 0.0030 |
| 6 | 7.134 | 3.7e+14 | 0.0063 |
| 8 | 6.756 | 6.9e+14 | 0.0116 |
| 10 | 6.620 | 1.1e+15 | 0.0187 |
| 12 | 6.433 | 1.5e+15 | 0.0258 |
| 14 | 6.266 | 2.2e+15 | 0.0366 |
| 16 | 6.070 | 2.9e+15 | 0.0481 |
| 18 | 6.057 | 3.7e+15 | 0.0627 |
| 20 | 6.033 | 4.5e+15 | 0.0759 |
| 22 | 5.905 | 5.4e+15 | 0.0908 |
| 24 | 5.826 | 6.4e+15 | 0.1073 |
| 26 | 5.674 | 7.5e+15 | 0.1255 |
| 28 | 5.742 | 8.6e+15 | 0.1453 |
| 30 | 5.614 | 9.9e+15 | 0.1669 |
| 32 | 5.532 | 1.1e+16 | 0.1931 |
| 34 | 5.357 | 1.3e+16 | 0.2181 |
| 36 | 5.316 | 1.5e+16 | 0.2483 |
| 38 | 5.345 | 1.6e+16 | 0.2770 |
| 40 | 5.245 | 1.9e+16 | 0.3113 |
| 42 | 5.170 | 2.1e+16 | 0.3477 |
| 46 | 5.090 | 2.3e+16 | 0.3822 |
| 50 | 5.170 | 2.5e+16 | 0.4206 |
| 54 | 5.094 | 2.8e+16 | 0.4631 |
| 58 | 5.015 | 3.0e+16 | 0.5097 |
| 62 | 4.876 | 3.3e+16 | 0.5605 |
| **66** | **4.843** | **3.7e+16** | **0.6156** |
| 70 | 5.049 | 4.0e+16 | 0.6752 |
| 74 | 5.019 | 4.4e+16 | 0.7394 |
| 78 | 5.116 | 4.8e+16 | 0.8083 |

Past `d = 66` the run drifts upward (~+0.2 val over thousands of steps);
the divergence guard in the trigger holds off further grows during the
drift, so the trajectory plateaus rather than blows up.

## Same-$d$ baseline

We train a fixed-$d{=}544$ E2E baseline from scratch with identical
training setup as the adaptive trajectory at its $d{=}544$ phase, and
run for matched wall-clock $6.81$h on one RTX PRO 6000 Blackwell.

| | best val CE | FLOPs to best | wall (h) |
|---|---:|---:|---:|
| Adaptive (d=2→496) | **3.565** | $1.00 \times 10^{18}$ | 6.81 |
| Fixed d=544 (matched wall) | 3.855 | $1.66 \times 10^{18}$ | 5.92 |

The adaptive run wins by $-0.29$ val CE while using $\approx
1.7{\times}$ fewer FLOPs to reach its best.  Past $d{=}544$ the
trigger plateaus near the Chinchilla floor for our $(N, D)$
($\approx 3.46$); further substantial val descent requires more
training tokens, not more model size.

See [`v11_vs_e2e.pdf`](v11_vs_e2e.pdf) (three-panel figure) for the
full comparison.

## v10 follow-up (fp32 + LR step-down)

Resumed the v8 trajectory at `d=46` (step 115 000) with the
precision/LR fixes from §3.5 and §6.

| precision / LR        | best val | at `d` | FLOP%  | notes                              |
|-----------------------|---------:|-------:|-------:|------------------------------------|
| `bf16 + AdamW8bit` 3e-4 | 4.843   | 66     | 0.62%  | v8 baseline                        |
| `fp32 + AdamW` 3e-4     | drifts  | 46     | 0.42%  | descends to 5.10, then up at d=50 |
| `fp32 + AdamW` 1.5e-4   | 5.041   | 46     | 0.42%  | LR/2 applied at resume             |
| `fp32 + AdamW` 7.5e-5   | 4.931   | 66     | 0.81%  | LR/4 applied at d=50; bf16 fwd     |

v10 reaches 4.93 at d=66 without the upward drift v8 had post-d=66, but
the lower LR makes the trajectory FLOP-inefficient relative to v8 below
d=66.  The v11 run resolves this by enabling dynamic `d_head` (no fixed
2) so that the attention memory ceiling stops binding past d≈40, which
lets the run stay in `bf16` forward (faster + fits more layers).

## FLOP-efficiency comparison

Best val attainable at or below a given FLOP fraction of the 5.95 PFLOPs
budget.

| FLOP% | E2E fixed (d=2048) | Adaptive | Adaptive `d` |
|---:|---:|---:|---:|
| 0.01 | — | 7.13 | 6 |
| 0.05 | — | 6.07 | 16 |
| 0.1 | — | 5.77 | 22 |
| 0.5 | — | 5.04 | 58 |
| 1.0 | — | 4.84 | 66 |
| 2.0 | 13.61 | 4.84 | 66 |
| 5.0 | 8.95 | 4.84 | 66 |
| 10.0 | 7.34 | 4.84 | 66 |
| 20.0 | 6.75 | 4.84 | 66 |
| 50.0 | 5.91 | 4.84 | 66 |
| 100.0 | 4.97 | 4.84 | 66 |

The adaptive run reaches the E2E end-point val of 4.97 at FLOP 0.62%; the
E2E baseline reaches the same val only at FLOP 100%.  Per-FLOP efficiency
at this loss is ~160×.

## E2E baseline is not at plateau

| step bucket | E2E best val | Δ |
|---:|---:|---:|
| 0 – 5 000 | 6.82 | — |
| 5 000 – 10 000 | 6.26 | -0.56 |
| 10 000 – 15 000 | 5.76 | -0.50 |
| 15 000 – 20 000 | 5.39 | -0.37 |
| 20 000 – 25 000 | 4.97 | -0.42 |

The final-bucket descent rate is still substantial (-0.42 per 5 000
steps).  A converged baseline would reach a noticeably lower val.  We
therefore report FLOP-efficiency, not absolute superiority.
