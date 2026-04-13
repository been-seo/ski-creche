import sqlite3
import sys
import json
from datetime import datetime


def status(db_path, run=None):
    conn = sqlite3.connect(db_path)

    # check schema
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if 'config' not in tables:
        _status_legacy(conn)
        conn.close()
        return

    runs = [r[0] for r in conn.execute(
        'SELECT DISTINCT run FROM config ORDER BY run').fetchall()]
    if run is None:
        run = runs[-1]

    print(f'run: {run}')
    print()

    # config
    rows = conn.execute(
        'SELECT key, value FROM config WHERE run=? ORDER BY key', (run,)
    ).fetchall()
    if rows:
        print('config:')
        for k, v in rows:
            print(f'  {k} = {v}')
        print()

    # phases
    rows = conn.execute(
        'SELECT K, start_step, start_flops FROM phases WHERE run=? ORDER BY K',
        (run,)
    ).fetchall()
    if rows:
        print('phases:')
        for K, s, f in rows:
            print(f'  K={K}  step={s}  flops={f:.3e}')
        print()

    # latest step
    row = conn.execute(
        'SELECT step, K, ce, lr, flops, gates, per_depth FROM train_log '
        'WHERE run=? ORDER BY step DESC LIMIT 1', (run,)
    ).fetchone()
    if row:
        step, K, ce, lr, flops, gates_s, pd_s = row
        gates = json.loads(gates_s) if gates_s else []
        per_depth = json.loads(pd_s) if pd_s else []
        print(f'latest: step={step}  K={K}  CE={ce:.4f}  lr={lr:.2e}  flops={flops:.3e}')
        if gates:
            print(f'  gates: [{", ".join(f"{g:.3f}" for g in gates)}]')
        if per_depth:
            print(f'  per_depth: [{", ".join(f"{d:.4f}" for d in per_depth)}]')
        print()

    # eval history
    rows = conn.execute(
        'SELECT step, K, val_ce, flops FROM eval_log '
        'WHERE run=? ORDER BY step', (run,)
    ).fetchall()
    if rows:
        print('eval:')
        best_ce = float('inf')
        for s, K, ce, f in rows:
            tag = ''
            if ce < best_ce:
                best_ce = ce
                tag = '  *best'
            print(f'  step={s:>5}  K={K}  CE={ce:.4f}  flops={f:.3e}{tag}')
        print()

    # checkpoints
    rows = conn.execute(
        'SELECT step, K, path FROM checkpoints WHERE run=? ORDER BY step',
        (run,)
    ).fetchall()
    if rows:
        print(f'checkpoints: ({len(rows)})')
        for s, K, p in rows:
            print(f'  step={s}  K={K}  {p}')
        print()

    # summary
    rows = conn.execute(
        'SELECT key, value FROM summary WHERE run=? ORDER BY key', (run,)
    ).fetchall()
    if rows:
        print('summary:')
        for k, v in rows:
            print(f'  {k} = {v}')

    conn.close()


def _status_legacy(conn):
    """Old schema with mode column instead of run."""
    print('(legacy schema)')
    print()
    for row in conn.execute(
        'SELECT mode, key, value FROM summary ORDER BY mode, key'
    ).fetchall():
        print(f'  {row[0]:>12} | {row[1]:>15} = {row[2]}')
    print()
    rows = conn.execute(
        'SELECT mode, step, K, val_ce, flops_cum FROM eval_log ORDER BY mode, step'
    ).fetchall()
    if rows:
        print('eval:')
        for mode, s, K, ce, f in rows:
            print(f'  {mode:>12}  step={s:>5}  K={K}  CE={ce:.4f}  flops={f:.3e}')


def compare(db_path):
    conn = sqlite3.connect(db_path)
    runs = [r[0] for r in conn.execute(
        'SELECT DISTINCT run FROM summary ORDER BY run').fetchall()]
    if not runs:
        _status_legacy(conn)
        conn.close()
        return

    print(f'{"run":>15} {"best_loss":>10} {"params":>12} {"total_flops":>14} {"steps":>7} {"elapsed":>8}')
    print('-' * 72)
    for run in runs:
        vals = {}
        for k, v in conn.execute(
            'SELECT key, value FROM summary WHERE run=?', (run,)
        ).fetchall():
            vals[k] = v
        print(f'{run:>15} '
              f'{float(vals.get("best_loss", "nan")):>10.4f} '
              f'{vals.get("params", "?"):>12} '
              f'{vals.get("total_flops", "?"):>14} '
              f'{vals.get("total_steps", "?"):>7} '
              f'{vals.get("elapsed", "?"):>8}')
    conn.close()


def main():
    if len(sys.argv) < 2:
        print('usage: python -m ski_creche <db_path> [--run NAME] [--compare]')
        sys.exit(1)

    db_path = sys.argv[1]
    if '--compare' in sys.argv:
        compare(db_path)
    else:
        run = None
        if '--run' in sys.argv:
            idx = sys.argv.index('--run')
            run = sys.argv[idx + 1]
        status(db_path, run)
