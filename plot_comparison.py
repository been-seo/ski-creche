"""3-panel academic figure: Adaptive (d=2 -> 496) vs Fixed d=544 E2E."""
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BATCH = 64
SEQ_LEN = 512
CHINCHILLA_FLOOR = 3.46  # for our (N=105M, D=2.9B) per Hoffmann et al.

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
    step = [r[0] for r in rows]
    d    = [r[1] for r in rows]
    val  = [r[2] for r in rows]
    flop = [r[3] for r in rows]
    ts   = [r[4] for r in rows]
    t0   = ts[0]
    hour = [(t - t0) / 3600 for t in ts]
    best, bv = [], float('inf')
    for v in val:
        if v < bv: bv = v
        best.append(bv)
    grow = [i for i in range(1, len(d)) if d[i] != d[i-1]]
    return dict(step=step, d=d, val=val, flop=flop, hour=hour,
                best=best, grow=grow)


adapt = load('/content/drive/MyDrive/ski/adaptive_v11.db')
e2e   = load('/content/drive/MyDrive/ski/e2e_d544.db')

C_ADAPT = '#1f77b4'   # tableau blue
C_E2E   = '#d62728'   # tableau red
C_BEST  = '#2ca02c'   # tableau green for stars

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), constrained_layout=True)

# -------- Panel 1: val vs FLOPs --------
ax = axes[0]
ax.semilogx(adapt['flop'], adapt['val'],
            'o', ms=2.0, alpha=0.20, color=C_ADAPT, mec='none')
ax.semilogx(adapt['flop'], adapt['best'], '-', color=C_ADAPT, lw=1.6,
            label='Adaptive ($d{:}2{\\to}496$)')
if e2e:
    ax.semilogx(e2e['flop'], e2e['val'],
                'o', ms=2.0, alpha=0.20, color=C_E2E, mec='none')
    ax.semilogx(e2e['flop'], e2e['best'], '-', color=C_E2E, lw=1.6,
                label='Fixed $d{=}544$ E2E')
ax.axhline(CHINCHILLA_FLOOR, ls='--', color='0.4', lw=0.8,
           label=f'Chinchilla floor ({CHINCHILLA_FLOOR:.2f})')
ax.set_xlabel(r'training compute (FLOPs)')
ax.set_ylabel(r'validation cross-entropy (nats)')
ax.legend(loc='upper right', frameon=False)
ax.grid(True, which='both', alpha=0.25, lw=0.4)
ax.set_xlim(1e15, 2e18)

# -------- Panel 2: val vs wall-clock --------
ax = axes[1]
ax.plot(adapt['hour'], adapt['val'],
        'o', ms=2.0, alpha=0.20, color=C_ADAPT, mec='none')
ax.plot(adapt['hour'], adapt['best'], '-', color=C_ADAPT, lw=1.6,
        label='Adaptive')
if e2e:
    ax.plot(e2e['hour'], e2e['val'],
            'o', ms=2.0, alpha=0.20, color=C_E2E, mec='none')
    ax.plot(e2e['hour'], e2e['best'], '-', color=C_E2E, lw=1.6,
            label='Fixed $d{=}544$')
ax.axhline(CHINCHILLA_FLOOR, ls='--', color='0.4', lw=0.8)
ax.set_xlabel(r'wall-clock (h, single RTX PRO 6000 Blackwell)')
ax.set_ylabel(r'validation cross-entropy (nats)')
ax.legend(loc='upper right', frameon=False)
ax.grid(True, alpha=0.25, lw=0.4)

# -------- Panel 3: d trajectory --------
ax = axes[2]
adapt_tok = [s * BATCH * SEQ_LEN / 1e9 for s in adapt['step']]
ax.plot(adapt_tok, adapt['d'], '-', color=C_ADAPT, lw=1.6,
        label='Adaptive width')
for i in adapt['grow']:
    ax.plot(adapt_tok[i], adapt['d'][i], '^', color=C_ADAPT,
            ms=4, alpha=0.7, mec='none')
bi = adapt['val'].index(min(adapt['val']))
ax.plot(adapt_tok[bi], adapt['d'][bi], '*', color=C_BEST, ms=12,
        mec='black', mew=0.4, zorder=5,
        label=f"Adaptive best  (val {adapt['val'][bi]:.3f})")
if e2e:
    e2e_tok = [s * BATCH * SEQ_LEN / 1e9 for s in e2e['step']]
    ax.plot(e2e_tok, e2e['d'], '-', color=C_E2E, lw=1.6,
            label='Fixed $d{=}544$')
    ei = e2e['val'].index(min(e2e['val']))
    ax.plot(e2e_tok[ei], e2e['d'][ei], '*', color=C_BEST, ms=12,
            mec='black', mew=0.4, zorder=5,
            label=f"Fixed best  (val {e2e['val'][ei]:.3f})")
ax.set_xlabel(r'training tokens seen ($10^9$)')
ax.set_ylabel(r'hidden dim $d$')
ax.legend(loc='lower right', frameon=False)
ax.grid(True, alpha=0.25, lw=0.4)

# Remove top/right spines for cleaner look
for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

plt.savefig('/content/drive/MyDrive/ski/ski-creche/scaleup/adapt_vs_e2e.pdf')
plt.savefig('/content/drive/MyDrive/ski/ski-creche/scaleup/adapt_vs_e2e.png', dpi=200)
print('saved adapt_vs_e2e.{pdf,png}')
print(f"adapt best: val={min(adapt['val']):.4f} d={adapt['d'][bi]} "
      f"flop={adapt['flop'][bi]:.3e} wall={adapt['hour'][bi]:.2f}h")
if e2e:
    print(f"E2E best:   val={min(e2e['val']):.4f} d={e2e['d'][ei]} "
          f"flop={e2e['flop'][ei]:.3e} wall={e2e['hour'][ei]:.2f}h")
