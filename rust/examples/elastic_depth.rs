#![recursion_limit = "512"]

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::prelude::*;
use burn::record::{NamedMpkFileRecorder, FullPrecisionSettings};

use ski_creche::config::SnowballConfig;
use ski_creche::model::SnowmanConfig;

type B = Autodiff<Wgpu>;
type BI = Wgpu;

fn main() {
    let device = WgpuDevice::default();
    let vocab = 256usize;
    let d = 64usize;
    let d_ff = 256usize;
    let seq_len = 32usize;
    let batch = 16usize;

    let make_batch = |device: &<B as Backend>::Device| {
        let data = Tensor::<B, 2, Int>::random(
            [batch, seq_len + 1],
            burn::tensor::Distribution::Uniform(0.0, vocab as f64),
            device,
        );
        let x = data.clone().slice([0..batch, 0..seq_len]);
        let y = data.slice([0..batch, 1..seq_len + 1]);
        (x, y)
    };

    // ── Phase 1: Train K=1->4 ───────────────────────────────
    println!("============================================================");
    println!("  Phase 1: Snowball K=1->4");
    println!("============================================================");

    let model = SnowmanConfig::new(vocab, d, d_ff, seq_len, 4).init::<B>(&device);

    let cfg = SnowballConfig {
        k_max: 4,
        total_steps: Some(200),
        eval_interval: 100,
        log_interval: 50,
        eval_batches: 5,
        ..SnowballConfig::new(4, 3e-4, 0.01, 1.0)
    };

    let (model, result) = ski_creche::trainer::train(
        model, &cfg, &device, batch * seq_len, None,
        |dev| make_batch(dev),
        Some(&mut |dev| make_batch(dev)),
        Some(&mut |step, k, ce, lr, _| {
            println!("  [base] step={step} K={k} CE={ce:.4} lr={lr:.2e}");
        }),
        Some(&mut |step, k, ce, _| {
            println!("  [base] eval step={step} K={k} CE={ce:.4}");
        }),
        Some(&mut |old, new, step, _| {
            println!("  [base] GROW K={old}->{new} at step {step}");
        }),
    );
    println!("  base K=4: best={:.4} final={:.4}", result.best_loss, result.final_loss);

    // Save checkpoint
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file("elastic_ckpt_K4", &recorder).expect("save");
    println!("  saved: elastic_ckpt_K4");
    println!("  gates: {:?}", model.gate_values());

    // ── Phase 2a: Attach K=4->8 ─────────────────────────────
    println!("\n============================================================");
    println!("  Phase 2a: Attach K=4->8 (grow)");
    println!("============================================================");

    let mut model2 = SnowmanConfig::new(vocab, d, d_ff, seq_len, 4).init::<B>(&device);
    model2 = model2.load_file("elastic_ckpt_K4", &recorder, &device).expect("load");
    model2 = model2.resize_depth(8, &device);
    println!("  gates after resize: {:?}", model2.gate_values());

    let cfg2 = SnowballConfig {
        k_max: 8,
        start_k: 5,
        total_steps: Some(200),
        eval_interval: 100,
        log_interval: 50,
        eval_batches: 5,
        ..SnowballConfig::new(8, 3e-4, 0.01, 1.0)
    };

    let (model2, result2) = ski_creche::trainer::train(
        model2, &cfg2, &device, batch * seq_len, None,
        |dev| make_batch(dev),
        Some(&mut |dev| make_batch(dev)),
        Some(&mut |step, k, ce, lr, _| {
            println!("  [attach] step={step} K={k} CE={ce:.4} lr={lr:.2e}");
        }),
        Some(&mut |step, k, ce, _| {
            println!("  [attach] eval step={step} K={k} CE={ce:.4}");
        }),
        Some(&mut |old, new, step, _| {
            println!("  [attach] GROW K={old}->{new} at step {step}");
        }),
    );
    println!("  attach K=8: best={:.4} final={:.4}", result2.best_loss, result2.final_loss);

    // ── Phase 2b: Detach ─────────────────────────────────────
    println!("\n============================================================");
    println!("  Phase 2b: Detach K=4->1..4 (no retraining)");
    println!("============================================================");

    let model3: ski_creche::model::Snowman<BI> = SnowmanConfig::new(vocab, d, d_ff, seq_len, 4)
        .init::<BI>(&device);
    let model3 = model3.load_file("elastic_ckpt_K4", &recorder, &device).expect("load");
    println!("  gates: {:?}", model3.gate_values());

    for k in 1..=4 {
        let mut total = 0.0;
        for _ in 0..20 {
            let data = Tensor::<BI, 2, Int>::random(
                [batch, seq_len + 1],
                burn::tensor::Distribution::Uniform(0.0, vocab as f64),
                &device,
            );
            let x = data.clone().slice([0..batch, 0..seq_len]);
            let y = data.slice([0..batch, 1..seq_len + 1]);
            let logits = model3.forward(x, k);
            let [b, t, v] = logits.dims();
            let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(&device)
                .forward(logits.reshape([b * t, v]), y.reshape([b * t]));
            total += loss.into_scalar().elem::<f64>();
        }
        let ce = total / 20.0;
        println!("  eval K={k}: CE={ce:.4}");
    }

    // Summary
    println!("\n============================================================");
    println!("  SUMMARY");
    println!("============================================================");
    println!("  base   K=4: best={:.4}", result.best_loss);
    println!("  attach K=8: best={:.4}", result2.best_loss);
    println!("  detach: see per-K eval above (no retraining needed)");

    let _ = std::fs::remove_file("elastic_ckpt_K4.mpk");
    let _ = model2;
}
