#![recursion_limit = "512"]

//! Elastic width demo: train at d=32, grow to d=64, verify preservation, train more.

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
    let d_small = 32usize;
    let d_ff_small = 128usize;
    let d_large = 64usize;
    let d_ff_large = 256usize;
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

    let make_batch_inner = |device: &<BI as Backend>::Device| {
        let data = Tensor::<BI, 2, Int>::random(
            [batch, seq_len + 1],
            burn::tensor::Distribution::Uniform(0.0, vocab as f64),
            device,
        );
        let x = data.clone().slice([0..batch, 0..seq_len]);
        let y = data.slice([0..batch, 1..seq_len + 1]);
        (x, y)
    };

    // ── Phase 1: Train K=1->4 at d=32 ────────────────────────
    println!("============================================================");
    println!("  Phase 1: Snowball K=1->4, d={d_small}");
    println!("============================================================");

    let model = SnowmanConfig::new(vocab, d_small, d_ff_small, seq_len, 4).init::<B>(&device);
    println!("  params: {}", model.param_count());

    let cfg = SnowballConfig {
        k_max: 4,
        total_steps: Some(200),
        eval_interval: 100,
        log_interval: 100,
        eval_batches: 5,
        ..SnowballConfig::new(4, 3e-4, 0.01, 1.0)
    };

    let (model, result1) = ski_creche::trainer::train(
        model, &cfg, &device, batch * seq_len, None,
        |dev| make_batch(dev),
        Some(&mut |dev| make_batch(dev)),
        Some(&mut |step, k, ce, lr, _| {
            println!("  step={step} K={k} CE={ce:.4} lr={lr:.2e}");
        }),
        Some(&mut |step, k, ce, _| {
            println!("  eval step={step} K={k} CE={ce:.4}");
        }),
        Some(&mut |old, new, step, _| {
            println!("  GROW K={old}->{new} at step {step}");
        }),
    );
    println!("  d={d_small} K=4: best={:.4}", result1.best_loss);

    // ── Verify preservation before/after width grow ───────────
    // Use inner backend (no autodiff) for clean comparison
    println!("\n============================================================");
    println!("  Width grow: d={d_small} -> d={d_large}");
    println!("============================================================");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file("width_ckpt", &recorder).expect("save");

    // Load as inner backend for comparison
    let model_small: ski_creche::model::Snowman<BI> =
        SnowmanConfig::new(vocab, d_small, d_ff_small, seq_len, 4)
            .init::<BI>(&device);
    let model_small = model_small.load_file("width_ckpt", &recorder, &device).expect("load");

    let (x_test, _y_test) = make_batch_inner(&device);
    let logits_before = model_small.forward(x_test.clone(), 4);

    // Grow width on inner backend model (d_active stays at old_d for preservation)
    let model_large_inner = model_small.grow_width(d_large, d_ff_large, &device);
    println!("  params after grow: {}", model_large_inner.param_count());
    println!("  d: {d_small} -> {}", model_large_inner.d);

    let logits_after = model_large_inner.forward(x_test.clone(), 4);

    let diff = (logits_before - logits_after).abs();
    let max_err: f32 = diff.max().into_scalar().elem();
    println!("  max |logits_before - logits_after| = {max_err:.2e}");
    if max_err < 1e-4 {
        println!("  [OK] Width preservation verified");
    } else {
        println!("  [FAIL] Width preservation broken: {max_err}");
    }

    // Now activate new width and save for Phase 2 training
    let mut model_large_inner = model_large_inner;
    model_large_inner.activate_width();
    model_large_inner.save_file("width_ckpt_grown", &recorder).expect("save");

    // Load as autodiff for training (init with large dims so skip fields match)
    let model_ad: ski_creche::model::Snowman<B> =
        SnowmanConfig::new(vocab, d_large, d_ff_large, seq_len, 4)
            .init::<B>(&device);
    let model_ad = model_ad.load_file("width_ckpt_grown", &recorder, &device).expect("load");

    // ── Phase 2: Train at d=64, K=4 fixed ────────────────────
    println!("\n============================================================");
    println!("  Phase 2: Train at d={d_large}, K=4 fixed");
    println!("============================================================");

    let cfg2 = SnowballConfig {
        k_max: 4,
        start_k: 4,
        total_steps: Some(200),
        eval_interval: 100,
        log_interval: 100,
        eval_batches: 5,
        ..SnowballConfig::new(4, 3e-4, 0.01, 1.0)
    };

    let (model_ad, result2) = ski_creche::trainer::train(
        model_ad, &cfg2, &device, batch * seq_len, None,
        |dev| make_batch(dev),
        Some(&mut |dev| make_batch(dev)),
        Some(&mut |step, k, ce, lr, _| {
            println!("  step={step} K={k} CE={ce:.4} lr={lr:.2e}");
        }),
        Some(&mut |step, k, ce, _| {
            println!("  eval step={step} K={k} CE={ce:.4}");
        }),
        None,
    );
    println!("  d={d_large} K=4: best={:.4}", result2.best_loss);

    // ── Depth detach test (inner backend) ─────────────────────
    println!("\n============================================================");
    println!("  Depth detach at d={d_large}");
    println!("============================================================");

    model_ad.clone().save_file("width_ckpt_large", &recorder).expect("save");
    let model_eval: ski_creche::model::Snowman<BI> =
        SnowmanConfig::new(vocab, d_large, d_ff_large, seq_len, 4)
            .init::<BI>(&device);
    let model_eval = model_eval.load_file("width_ckpt_large", &recorder, &device).expect("load");

    for k in 1..=4 {
        let mut total = 0.0;
        for _ in 0..10 {
            let (x, y) = make_batch_inner(&device);
            let logits = model_eval.forward(x, k);
            let [b, t, v] = logits.dims();
            let loss = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(&device)
                .forward(logits.reshape([b * t, v]), y.reshape([b * t]));
            total += loss.into_scalar().elem::<f64>();
        }
        println!("  K={k}: CE={:.4}", total / 10.0);
    }

    // Cleanup
    let _ = std::fs::remove_file("width_ckpt.mpk");
    let _ = std::fs::remove_file("width_ckpt_grown.mpk");
    let _ = std::fs::remove_file("width_ckpt_large.mpk");

    // ── Summary ───────────────────────────────────────────────
    println!("\n============================================================");
    println!("  SUMMARY");
    println!("============================================================");
    println!("  d={d_small} K=4: best={:.4}", result1.best_loss);
    println!("  d={d_large} K=4 (grown): best={:.4}", result2.best_loss);
    println!("  Width preservation: max_err={max_err:.2e}");
}
