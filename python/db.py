import sqlite3
import time
import json


class TrainLogger:
    def __init__(self, db_path, config=None):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS config (
            run TEXT, key TEXT, value TEXT, PRIMARY KEY(run, key))''')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS train_log (
            run TEXT, step INTEGER, K INTEGER,
            ce REAL, lr REAL, flops REAL,
            gates TEXT, per_depth TEXT, timestamp REAL,
            PRIMARY KEY(run, step))''')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS eval_log (
            run TEXT, step INTEGER, K INTEGER,
            val_ce REAL, flops REAL, timestamp REAL,
            PRIMARY KEY(run, step, K))''')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS phases (
            run TEXT, K INTEGER, start_step INTEGER,
            start_flops REAL, timestamp REAL,
            PRIMARY KEY(run, K))''')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS checkpoints (
            run TEXT, step INTEGER, K INTEGER,
            path TEXT, timestamp REAL,
            PRIMARY KEY(run, step))''')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS summary (
            run TEXT, key TEXT, value TEXT,
            PRIMARY KEY(run, key))''')
        self.conn.commit()

        if config is not None:
            self._save_config(config)

    def _save_config(self, config):
        run = config.run_name
        for name in config.__dataclass_fields__:
            val = getattr(config, name)
            if callable(val):
                continue
            self.conn.execute('INSERT OR REPLACE INTO config VALUES(?,?,?)',
                              (run, name, str(val)))
        self.conn.commit()

    def log_step(self, run, step, K, ce, lr, flops, gates, per_depth):
        self.conn.execute(
            'INSERT OR REPLACE INTO train_log VALUES(?,?,?,?,?,?,?,?,?)',
            (run, step, K, ce, lr, flops,
             json.dumps(gates), json.dumps(per_depth), time.time()))

    def log_eval(self, run, step, K, val_ce, flops):
        self.conn.execute(
            'INSERT OR REPLACE INTO eval_log VALUES(?,?,?,?,?,?)',
            (run, step, K, val_ce, flops, time.time()))

    def log_phase(self, run, K, step, flops):
        self.conn.execute(
            'INSERT OR REPLACE INTO phases VALUES(?,?,?,?,?)',
            (run, K, step, flops, time.time()))

    def log_checkpoint(self, run, step, K, path):
        self.conn.execute(
            'INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?,?)',
            (run, step, K, path, time.time()))

    def log_summary(self, run, key, value):
        self.conn.execute(
            'INSERT OR REPLACE INTO summary VALUES(?,?,?)',
            (run, key, str(value)))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()
