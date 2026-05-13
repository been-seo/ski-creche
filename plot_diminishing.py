"""Plot v11 val vs d (log-log) showing diminishing returns past d~100."""
import sqlite3, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

c = sqlite3.connect(__import__('os').environ.get('ADAPT_DB', 'adaptive.db'))
rows = c.execute("""
SELECT d, MIN(val_ce), MAX(flops_used), AVG(val_ce) FROM log WHERE val_ce > 0
GROUP BY d ORDER BY d""").fetchall()
d  = [r[0] for r in rows]
v  = [r[1] for r in rows]
fr = [r[2]/5.95e18*100 for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.0))

ax[0].semilogx(d, v, 'o-', markersize=3, color='steelblue')
ax[0].axhline(4.84, ls=':', color='gray', label='v8 best (d=66)')
ax[0].axhline(4.97, ls='--', color='red', label='E2E final (FLOP 100\\%)')
ax[0].set_xlabel('width $d$')
ax[0].set_ylabel('best val CE at width $d$')
ax[0].legend(loc='upper right', fontsize=8)
ax[0].grid(alpha=0.3)

# val vs FLOP
ax[1].semilogx(fr, v, 'o-', markersize=3, color='darkorange')
ax[1].axhline(4.84, ls=':', color='gray', label='v8 best')
ax[1].axhline(4.97, ls='--', color='red', label='E2E final (FLOP 100\\%)')
ax[1].set_xlabel('FLOP usage (\\%)')
ax[1].set_ylabel('best val CE')
ax[1].legend(loc='upper right', fontsize=8)
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('paper/v11_diminishing.pdf')
print('Saved v11_diminishing.pdf')
