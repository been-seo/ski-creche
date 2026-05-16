"""
Parse adaptive_growth log; estimate noise distribution on stationary segments.

Pipeline:
  1. parse eval lines; tag recov/ramp/DIV.
  2. per-d stationary segments (excluding recov/ramp).
  3. linear detrend → residuals → σ̂ per d.
  4. pooled σ + Gaussianity tests (Shapiro, Anderson-Darling, kurtosis, Student-t fit).
  5. integration detector: lag-1 autocorrelation, variance-growth at lags 1/2/3.
     For pure i.i.d., Var(r_{t+k}−r_t) is constant in k.  For integrated random
     walk, ∝ k.  Slope-1 growth ⇒ σ_int ∝ √T.
  6. standardized residuals (r/σ_d) to disentangle σ-heterogeneity from shape.
"""
import re, sys
import numpy as np
from scipy import stats

LOG = sys.argv[1] if len(sys.argv) > 1 else 'adaptive.log'

eval_re = re.compile(r'eval (\d+) d=(\d+) val=([\d.]+) best=([\d.]+) prog=([+\-\d.NA]+) d_eff_W=([\d.]+)(?:\s*\[recov\])?(?:\s*\[ramp [^\]]+\])?')
grow_re = re.compile(r'\*\*\* GROW d=(\d+)→(\d+)')

events = []
grow_events = []
with open(LOG) as f:
    for line in f:
        if 'eval' in line and 'val=' in line:
            m = eval_re.search(line)
            if m:
                step, d, val = int(m.group(1)), int(m.group(2)), float(m.group(3))
                tags = []
                if '[recov]' in line: tags.append('recov')
                if '[ramp' in line: tags.append('ramp')
                if '[DIV]' in line: tags.append('DIV')
                events.append((step, d, val, tags))
        elif 'GROW' in line:
            m = grow_re.search(line)
            if m:
                grow_events.append(events[-1][0] if events else 0)

print(f'Total evals: {len(events)}')
print(f'Grow events: {len(grow_events)}: {grow_events}')

segments = {}
for step, d, val, tags in events:
    if 'recov' in tags or 'ramp' in tags:
        continue
    segments.setdefault(d, []).append((step, val))

print('\n=== Per-d stationary segments ===')
all_residuals = []
for d in sorted(segments.keys()):
    seg = segments[d]
    if len(seg) < 3:
        print(f'  d={d}: {len(seg)} evals — skipping (too short)')
        continue
    t = np.array([s[0] for s in seg], dtype=float)
    v = np.array([s[1] for s in seg], dtype=float)
    coef = np.polyfit(t, v, 1)
    trend = np.polyval(coef, t)
    r = v - trend
    sigma = r.std(ddof=1) if len(r) > 1 else 0.0
    print(f'  d={d}: N={len(seg)}, slope={coef[0]*1000:.4f}/1k_steps, σ̂={sigma:.4f}')
    all_residuals.extend(r.tolist())

print(f'\n=== Pooled stationary residuals (N={len(all_residuals)}) ===')
r = np.array(all_residuals)
if len(r) >= 5:
    print(f'  σ̂      = {r.std(ddof=1):.4f}')
    print(f'  mean   = {r.mean():.4f} (expect 0)')
    print(f'  skew   = {stats.skew(r):.4f} (Gaussian=0)')
    print(f'  kurt   = {stats.kurtosis(r, fisher=False):.4f} (Gaussian=3)')
    print(f'  range  = [{r.min():.4f}, {r.max():.4f}]')
    if len(r) >= 8:
        sh = stats.shapiro(r)
        print(f'  Shapiro-Wilk: W={sh.statistic:.4f} p={sh.pvalue:.4f}  '
              f'({"Gaussian likely" if sh.pvalue>0.05 else "non-Gaussian"})')
    if len(r) >= 7:
        ad = stats.anderson(r, dist='norm')
        sig5 = ad.critical_values[2]
        print(f'  Anderson-Darling: A={ad.statistic:.4f} crit5%={sig5:.4f}  '
              f'({"Gaussian likely" if ad.statistic<sig5 else "non-Gaussian"})')
    if len(r) >= 10:
        df_t, _, scale_t = stats.t.fit(r, floc=0)
        print(f'  Student-t fit: df={df_t:.2f} scale={scale_t:.4f}  '
              f'(df→∞ ≈ Gaussian; df<10 = heavy-tail)')

print('\n=== DIV events ===')
for step, d, val, tags in events:
    if 'DIV' in tags:
        print(f'  step {step} d={d} val={val:.4f}')

if len(r) >= 12:
    half = len(r) // 2
    s1, s2 = r[:half].std(ddof=1), r[half:].std(ddof=1)
    print(f'\n=== σ stationarity (1st half vs 2nd half) ===')
    print(f'  σ_1st  = {s1:.4f}  (N={half})')
    print(f'  σ_2nd  = {s2:.4f}  (N={len(r)-half})')
    print(f'  ratio  = {s2/s1:.3f}  (>1 = growing noise → integration suspected)')

print('\n=== Lag-1 autocorrelation (per-d) — integration detector ===')
print('  Var(Δr)/Var(r) = 2(1−ρ_1); ≈2 iid, →0 integrated, >2 anti-corr')
for d in sorted(segments.keys()):
    seg = segments[d]
    if len(seg) < 4:
        continue
    t = np.array([s[0] for s in seg], dtype=float)
    v = np.array([s[1] for s in seg], dtype=float)
    coef = np.polyfit(t, v, 1)
    rr = v - np.polyval(coef, t)
    if rr.std(ddof=1) < 1e-6:
        continue
    rho1 = np.corrcoef(rr[:-1], rr[1:])[0, 1] if len(rr) >= 3 else float('nan')
    dr = np.diff(rr)
    ratio = dr.var(ddof=1) / rr.var(ddof=1) if rr.var(ddof=1) > 0 else float('nan')
    print(f'  d={d:>3}: N={len(seg)}, ρ_1={rho1:+.3f}, Var(Δr)/Var(r)={ratio:.2f}')

print('\n=== Variance-growth test (lag scaling) ===')
print('  k    | <Var(r_{t+k}-r_t)> | iid pred | int pred')
all_lagvar = {1: [], 2: [], 3: []}
for d, seg in segments.items():
    if len(seg) < 5:
        continue
    t = np.array([s[0] for s in seg], dtype=float)
    v = np.array([s[1] for s in seg], dtype=float)
    coef = np.polyfit(t, v, 1)
    rr = v - np.polyval(coef, t)
    sig = rr.std(ddof=1)
    if sig < 1e-6:
        continue
    rr_n = rr / sig
    for k in [1, 2, 3]:
        if len(rr_n) > k:
            diffs = rr_n[k:] - rr_n[:-k]
            all_lagvar[k].extend((diffs**2).tolist())
for k in [1, 2, 3]:
    if all_lagvar[k]:
        mean_var = np.mean(all_lagvar[k])
        print(f'  k={k}  | {mean_var:.3f}              | 2.00     | {2*k}.00')

print('\n=== Standardized residuals (r/σ_d, removes σ-heterogeneity) ===')
zs = []
for d, seg in segments.items():
    if len(seg) < 3:
        continue
    t = np.array([s[0] for s in seg], dtype=float)
    v = np.array([s[1] for s in seg], dtype=float)
    coef = np.polyfit(t, v, 1)
    rr = v - np.polyval(coef, t)
    sig = rr.std(ddof=1)
    if sig > 1e-6:
        zs.extend((rr / sig).tolist())
z = np.array(zs)
if len(z) >= 8:
    print(f'  N      = {len(z)}')
    print(f'  σ̂(z)   = {z.std(ddof=1):.4f}  (expect 1.0 by construction)')
    print(f'  skew   = {stats.skew(z):.4f}')
    print(f'  kurt   = {stats.kurtosis(z, fisher=False):.4f}  (Gaussian=3, heavy-tail>3)')
    sh = stats.shapiro(z)
    print(f'  Shapiro-Wilk: W={sh.statistic:.4f} p={sh.pvalue:.4f}  '
          f'({"Gaussian likely" if sh.pvalue>0.05 else "non-Gaussian"})')
    ad = stats.anderson(z, dist='norm')
    sig5 = ad.critical_values[2]
    print(f'  Anderson-Darling: A={ad.statistic:.4f} crit5%={sig5:.4f}  '
          f'({"Gaussian likely" if ad.statistic<sig5 else "non-Gaussian"})')
    df_t, _, sc_t = stats.t.fit(z, floc=0)
    print(f'  Student-t fit: df={df_t:.2f} scale={sc_t:.4f}')
