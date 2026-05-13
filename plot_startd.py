"""Start-d ablation: adaptive d=2 start vs adaptive d=64 start vs fixed-d=544 E2E."""
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHINCHILLA_FLOOR = 3.46

plt.rcParams.update({
    'font.family':     'serif',
    'font.serif':      ['Times', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset':'cm',
    'font.size':       9,
    'axes.labelsize':  9,
    'axes.titlesize':  10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth':  0.6,
    'lines.linewidth': 1.5,
    'lines.markersize':3,
    'figure.dpi':      150,
})

def load(path):
    c = sqlite3.connect(path)
    rows = c.execute(
        'SELECT step, d, val_ce, flops_used, ts FROM log '
        'WHERE val_ce > 0 ORDER BY step'
    ).fetchall()
    if not rows:
        return None
    val  = [r[2] for r in rows]
    flop = [r[3] for r in rows]
    ts   = [r[4] for r in rows]
    t0   = ts[0]
    hour = [(t - t0) / 3600 for t in ts]
    best, bv = [], float('inf')
    for v in val:
        if v < bv: bv = v
        best.append(bv)
    return dict(val=val, flop=flop, hour=hour, best=best)

import os
adapt_d2  = load(os.environ.get('ADAPT_D2_DB',  'adaptive.db'))
adapt_d64 = load(os.environ.get('ADAPT_D64_DB', 'adaptive_startd64.db'))
e2e_d544  = load(os.environ.get('E2E_DB',       'e2e_d544.db'))

C_D2  = '#1f77b4'  # blue
C_D64 = '#9467bd'  # purple
C_E2E = '#d62728'  # red

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), constrained_layout=True)

# Panel 1: val vs FLOPs
ax = axes[0]
ax.semilogx(adapt_d2['flop'],  adapt_d2['best'],  '-', color=C_D2, lw=1.7,
            label='Adaptive start $d{=}2$')
ax.semilogx(adapt_d64['flop'], adapt_d64['best'], '-', color=C_D64, lw=1.7,
            label='Adaptive start $d{=}64$')
if e2e_d544:
    ax.semilogx(e2e_d544['flop'], e2e_d544['best'], '-', color=C_E2E, lw=1.7,
                label='Fixed $d{=}544$ E2E')
ax.axhline(CHINCHILLA_FLOOR, ls='--', color='0.4', lw=0.8,
           label=f'Chinchilla floor ({CHINCHILLA_FLOOR:.2f})')
ax.set_xlabel(r'training compute (FLOPs)')
ax.set_ylabel(r'best validation CE (nats)')
ax.legend(loc='upper right', frameon=False)
ax.grid(True, which='both', alpha=0.25, lw=0.4)
ax.set_xlim(1e15, 2e18)
ax.set_ylim(3.3, 6.5)

# Panel 2: val vs wall-clock
ax = axes[1]
ax.plot(adapt_d2['hour'],  adapt_d2['best'],  '-', color=C_D2, lw=1.7,
        label='Adaptive start $d{=}2$')
ax.plot(adapt_d64['hour'], adapt_d64['best'], '-', color=C_D64, lw=1.7,
        label='Adaptive start $d{=}64$')
if e2e_d544:
    ax.plot(e2e_d544['hour'], e2e_d544['best'], '-', color=C_E2E, lw=1.7,
            label='Fixed $d{=}544$ E2E')
ax.axhline(CHINCHILLA_FLOOR, ls='--', color='0.4', lw=0.8)
ax.set_xlabel(r'wall-clock (h, single RTX PRO 6000 Blackwell)')
ax.set_ylabel(r'best validation CE (nats)')
ax.legend(loc='upper right', frameon=False)
ax.grid(True, alpha=0.25, lw=0.4)
ax.set_xlim(0, 8)
ax.set_ylim(3.3, 6.5)

for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.savefig('paper/startd_ablation.pdf')
plt.savefig('paper/startd_ablation.png', dpi=200)
print('saved startd_ablation.{pdf,png}')

for name, r in [('d=2', adapt_d2), ('d=64', adapt_d64), ('E2E', e2e_d544)]:
    if r:
        i = r['val'].index(min(r['val']))
        print(f'{name:6s} best: val={r["val"][i]:.4f} flop={r["flop"][i]:.3e} wall={r["hour"][i]:.2f}h')
