use rusqlite::{params, Connection};
use std::time::{SystemTime, UNIX_EPOCH};

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

pub struct TrainLogger {
    conn: Connection,
}

impl TrainLogger {
    pub fn open(path: &str) -> Self {
        let conn = Connection::open(path).expect("failed to open db");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS config (
                run TEXT, key TEXT, value TEXT, PRIMARY KEY(run, key));
            CREATE TABLE IF NOT EXISTS train_log (
                run TEXT, step INTEGER, K INTEGER,
                ce REAL, lr REAL, flops REAL,
                gates TEXT, per_depth TEXT, timestamp REAL,
                PRIMARY KEY(run, step));
            CREATE TABLE IF NOT EXISTS eval_log (
                run TEXT, step INTEGER, K INTEGER,
                val_ce REAL, flops REAL, timestamp REAL,
                PRIMARY KEY(run, step, K));
            CREATE TABLE IF NOT EXISTS phases (
                run TEXT, K INTEGER, start_step INTEGER,
                start_flops REAL, timestamp REAL,
                PRIMARY KEY(run, K));
            CREATE TABLE IF NOT EXISTS checkpoints (
                run TEXT, step INTEGER, K INTEGER,
                path TEXT, timestamp REAL,
                PRIMARY KEY(run, step));
            CREATE TABLE IF NOT EXISTS summary (
                run TEXT, key TEXT, value TEXT,
                PRIMARY KEY(run, key));",
        )
        .expect("failed to create tables");
        TrainLogger { conn }
    }

    pub fn log_config(&self, run: &str, key: &str, value: &str) {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO config VALUES(?1,?2,?3)",
                params![run, key, value],
            )
            .ok();
    }

    pub fn log_step(
        &self, run: &str, step: usize, k: usize, ce: f64, lr: f64, flops: f64,
        gates: &str, per_depth: &str,
    ) {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO train_log VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![run, step as i64, k as i64, ce, lr, flops, gates, per_depth, now()],
            )
            .ok();
    }

    pub fn log_eval(&self, run: &str, step: usize, k: usize, val_ce: f64, flops: f64) {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO eval_log VALUES(?1,?2,?3,?4,?5,?6)",
                params![run, step as i64, k as i64, val_ce, flops, now()],
            )
            .ok();
    }

    pub fn log_phase(&self, run: &str, k: usize, step: usize, flops: f64) {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO phases VALUES(?1,?2,?3,?4,?5)",
                params![run, k as i64, step as i64, flops, now()],
            )
            .ok();
    }

    pub fn log_checkpoint(&self, run: &str, step: usize, k: usize, path: &str) {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO checkpoints VALUES(?1,?2,?3,?4,?5)",
                params![run, step as i64, k as i64, path, now()],
            )
            .ok();
    }

    pub fn log_summary(&self, run: &str, key: &str, value: &str) {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO summary VALUES(?1,?2,?3)",
                params![run, key, value],
            )
            .ok();
    }

    pub fn commit(&self) {
        // rusqlite auto-commits by default; explicit transaction control if needed
    }
}
