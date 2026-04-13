use std::time::Instant;

use burn::prelude::*;
// AutodiffModule is used implicitly via Module derive
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::grad_clipping::GradientClippingConfig;

use crate::config::SnowballConfig;
use crate::db::TrainLogger;
use crate::flops::flop_single;
use crate::model::Snowman;

pub struct TrainResult {
    pub best_loss: f64,
    pub final_loss: f64,
    pub total_steps: usize,
    pub elapsed: f64,
    pub total_flops: f64,
    pub params: usize,
    pub block_params: usize,
}

pub fn train<B>(
    mut model: Snowman<B>,
    cfg: &SnowballConfig,
    device: &B::Device,
    tokens_per_step: usize,
    flop_budget: Option<f64>,
    mut train_data: impl FnMut(&B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>),
    mut eval_data: Option<&mut dyn FnMut(&B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>)>,
    mut on_step: Option<&mut dyn FnMut(usize, usize, f64, f64, f64)>,
    mut on_eval: Option<&mut dyn FnMut(usize, usize, f64, f64)>,
    mut on_phase: Option<&mut dyn FnMut(usize, usize, usize, f64)>,
) -> (Snowman<B>, TrainResult)
where
    B: burn::tensor::backend::AutodiffBackend,
{
    B::seed(device, cfg.seed);

    let p_emb = model.embed_param_count();
    let b_block = model.block_param_count();
    let p_head = model.readout_param_count();

    let mut optim = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay as f32)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.grad_clip as f32)))
        .init::<B, Snowman<B>>();

    let logger = cfg.db_path.as_ref().map(|p| TrainLogger::open(p));

    let phase_total = if let Some(budget) = flop_budget {
        let total_cost: usize = (1..=cfg.k_max)
            .map(|k| flop_single(k, p_emb, b_block, p_head))
            .sum();
        (budget / (tokens_per_step as f64 * total_cost as f64)) as usize
    } else {
        cfg.total_steps.expect("need flop_budget or total_steps")
    };

    let t0 = Instant::now();
    let mut flops_cum: f64 = 0.0;
    let mut best_loss = f64::INFINITY;
    let mut step: usize = 0;
    let mut current_k: usize = cfg.start_k;
    let mut phase_step: usize = 0;
    let mut last_ce: f64 = 0.0;

    if let Some(ref lg) = logger {
        lg.log_phase(&cfg.run_name, current_k, step, flops_cum);
    }

    loop {
        if phase_step >= phase_total {
            if current_k >= cfg.k_max {
                break;
            }
            let old_k = current_k;
            current_k += 1;
            phase_step = 0;

            optim = AdamWConfig::new()
                .with_weight_decay(cfg.weight_decay as f32)
                .with_grad_clipping(Some(GradientClippingConfig::Norm(cfg.grad_clip as f32)))
                .init::<B, Snowman<B>>();

            if let Some(ref lg) = logger {
                lg.log_phase(&cfg.run_name, current_k, step, flops_cum);
            }
            if let Some(ref mut cb) = on_phase {
                cb(old_k, current_k, step, flops_cum);
            }
        }

        let lr = cosine_lr(cfg.lr, phase_step, phase_total);

        let (x, y) = train_data(device);

        // Random depth sampling: k ~ U(1..current_K)
        let k = (rand::random::<usize>() % current_k) + 1;

        let loss = model.forward_single(x, k, y);
        let ce = loss.clone().into_scalar().elem::<f64>();
        last_ce = ce;

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads_params);

        let fpt = flop_single(k, p_emb, b_block, p_head);
        flops_cum += fpt as f64 * tokens_per_step as f64;

        if step % cfg.log_interval == 0 {
            let gates = model.gate_values();
            let gates_json = serde_json::to_string(&gates[..current_k]).unwrap();
            let pd_json = serde_json::to_string(&[ce]).unwrap();
            if let Some(ref lg) = logger {
                lg.log_step(&cfg.run_name, step, current_k, ce, lr, flops_cum, &gates_json, &pd_json);
            }
            if let Some(ref mut cb) = on_step {
                cb(step, current_k, ce, lr, flops_cum);
            }
        }

        if (step + 1) % cfg.eval_interval == 0 {
            if let Some(ref mut ed) = eval_data {
                let mut total = 0.0;
                for _ in 0..cfg.eval_batches {
                    let (x, y) = ed(device);
                    // Use autodiff model but detach the result (no grad tracking needed)
                    let logits = model.forward(x, current_k).detach();
                    let [b, t, v] = logits.dims();
                    let logits_flat = logits.reshape([b * t, v]);
                    let targets_flat = y.reshape([b * t]);
                    let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                        .init(&logits_flat.device())
                        .forward(logits_flat, targets_flat);
                    total += loss.into_scalar().elem::<f64>();
                }
                let val_ce = total / cfg.eval_batches as f64;
                if val_ce < best_loss {
                    best_loss = val_ce;
                }
                if let Some(ref lg) = logger {
                    lg.log_eval(&cfg.run_name, step + 1, current_k, val_ce, flops_cum);
                }
                if let Some(ref mut cb) = on_eval {
                    cb(step + 1, current_k, val_ce, flops_cum);
                }
            }
        }

        step += 1;
        phase_step += 1;
    }

    let elapsed = t0.elapsed().as_secs_f64();

    let final_loss = if let Some(ref mut ed) = eval_data {
        let mut total = 0.0;
        for _ in 0..cfg.eval_batches {
            let (x, y) = ed(device);
            let logits = model.forward(x, cfg.k_max).detach();
            let [b, t, v] = logits.dims();
            let logits_flat = logits.reshape([b * t, v]);
            let targets_flat = y.reshape([b * t]);
            let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(&logits_flat.device())
                .forward(logits_flat, targets_flat);
            total += loss.into_scalar().elem::<f64>();
        }
        total / cfg.eval_batches as f64
    } else {
        last_ce
    };
    if final_loss < best_loss {
        best_loss = final_loss;
    }

    let result = TrainResult {
        best_loss,
        final_loss,
        total_steps: step,
        elapsed,
        total_flops: flops_cum,
        params: model.param_count(),
        block_params: b_block,
    };

    if let Some(ref lg) = logger {
        lg.log_summary(&cfg.run_name, "best_loss", &format!("{:.6}", result.best_loss));
        lg.log_summary(&cfg.run_name, "final_loss", &format!("{:.6}", result.final_loss));
        lg.log_summary(&cfg.run_name, "total_steps", &result.total_steps.to_string());
        lg.log_summary(&cfg.run_name, "elapsed", &format!("{:.1}", result.elapsed));
        lg.log_summary(&cfg.run_name, "total_flops", &format!("{:.3e}", result.total_flops));
        lg.log_summary(&cfg.run_name, "params", &result.params.to_string());
    }

    (model, result)
}

fn cosine_lr(base_lr: f64, phase_step: usize, phase_total: usize) -> f64 {
    base_lr * 0.5 * (1.0 + (std::f64::consts::PI * phase_step as f64 / phase_total as f64).cos())
}
