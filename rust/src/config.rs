pub struct SnowballConfig {
    pub k_max: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub total_steps: Option<usize>,
    pub eval_interval: usize,
    pub log_interval: usize,
    pub eval_batches: usize,
    pub seed: u64,
    pub start_k: usize,
    pub checkpoint_dir: Option<String>,
    pub checkpoint_interval: Option<usize>,
    pub checkpoint_on_phase: bool,
    pub checkpoint_best: bool,
    pub db_path: Option<String>,
    pub run_name: String,
}

impl SnowballConfig {
    pub fn new(k_max: usize, lr: f64, weight_decay: f64, grad_clip: f64) -> Self {
        Self {
            k_max,
            lr,
            weight_decay,
            grad_clip,
            start_k: 1,
            total_steps: None,
            eval_interval: 200,
            log_interval: 200,
            eval_batches: 10,
            seed: 42,
            checkpoint_dir: None,
            checkpoint_interval: None,
            checkpoint_on_phase: true,
            checkpoint_best: true,
            db_path: None,
            run_name: "snowball".into(),
        }
    }
}
