"""FLOP-matched comparison between an adaptive run and the E2E fixed baseline.

Prints best val CE achievable at fixed fractions of a target FLOP budget.

The adaptive DB is expected to log `flops_used` per eval; the E2E baseline DB
may or may not (we recompute from `step` and a fixed `d`).
"""
import sqlite3
import os

V, N_L, B_, T_ = 50257, 22, 64, 512
TARGET = float(os.environ.get('TARGET_FLOPS', 5.95e18))


def flop_per_step(d):
    return 6 * (V * d + N_L * 12 * d * d) * B_ * T_


def fetch_with_flops(db, run_filter=None, fixed_d=None):
    c = sqlite3.connect(db)
    cols = [r[1] for r in c.execute("PRAGMA table_info(log)").fetchall()]
    has_flops = 'flops_used' in cols
    cols_q = 'step, d, val_ce, flops_used' if has_flops else 'step, d, val_ce'
    q = f'SELECT {cols_q} FROM log WHERE val_ce > 0'
    if run_filter:
        q += f" AND run='{run_filter}'"
    q += ' ORDER BY step'
    rows = c.execute(q).fetchall()
    if has_flops:
        return list(rows)
    out = []
    for s, d, v in rows:
        d_use = fixed_d or d
        out.append((s, d_use, v, s * flop_per_step(d_use)))
    return out


def main(adaptive_db='adaptive.db', adaptive_run='adaptive',
         e2e_db=None, e2e_run='e2e_baseline', e2e_d=2048):
    adapt = fetch_with_flops(adaptive_db, adaptive_run)
    e2e = fetch_with_flops(e2e_db, e2e_run, fixed_d=e2e_d) if e2e_db else []

    print(f'{"FLOP%":>6} | {"E2E val":>8} {"E2E d":>6} | {"adapt val":>9} {"adapt d":>7}')
    print('-' * 56)
    for pct in [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]:
        target = TARGET * pct / 100
        e2e_at = [r for r in e2e if r[3] <= target]
        adapt_at = [r for r in adapt if r[3] <= target]
        e2e_best = min((r[2] for r in e2e_at), default=None)
        adapt_best = min((r[2] for r in adapt_at), default=None)
        e2e_d_now = e2e_at[-1][1] if e2e_at else '-'
        adapt_d_now = adapt_at[-1][1] if adapt_at else '-'
        e2e_str = f'{e2e_best:.3f}' if e2e_best is not None else '   -   '
        adapt_str = f'{adapt_best:.3f}' if adapt_best is not None else '    -    '
        print(f'{pct:>6.2f} | {e2e_str:>8} {str(e2e_d_now):>6} | '
              f'{adapt_str:>9} {str(adapt_d_now):>7}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--adaptive-db', default='adaptive.db')
    p.add_argument('--adaptive-run', default='adaptive')
    p.add_argument('--e2e-db', default=None)
    p.add_argument('--e2e-run', default='e2e_baseline')
    p.add_argument('--e2e-d', type=int, default=2048)
    args = p.parse_args()
    main(args.adaptive_db, args.adaptive_run, args.e2e_db, args.e2e_run, args.e2e_d)
