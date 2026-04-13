use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::prelude::*;

use ski_creche::config::SnowballConfig;
use ski_creche::flops::flop_ratio;
use ski_creche::model::SnowmanConfig;

type B = Autodiff<Wgpu>;

fn main() {
    let device = WgpuDevice::default();

    let vocab = 256;
    let d = 64;
    let d_ff = 256;
    let seq_len = 32;
    let batch = 16;
    let k_max = 4;

    let model = SnowmanConfig::new(vocab, d, d_ff, seq_len, k_max).init::<B>(&device);

    let ratio = flop_ratio(k_max, model.embed_param_count(), model.block_param_count(), model.readout_param_count());
    println!("params: {}, FLOP ratio: {:.3}", model.param_count(), ratio);

    let cfg = SnowballConfig {
        k_max,
        total_steps: Some(100),
        eval_interval: 50,
        log_interval: 25,
        eval_batches: 5,
        ..SnowballConfig::new(k_max, 3e-4, 0.01, 1.0)
    };

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

    let (model, result) = ski_creche::trainer::train(
        model,
        &cfg,
        &device,
        batch * seq_len,
        None,
        |dev| make_batch(dev),
        Some(&mut |dev| make_batch(dev)),
        Some(&mut |step, k, ce, lr, _flops| {
            println!("step={step} K={k} CE={ce:.4} lr={lr:.2e}");
        }),
        Some(&mut |step, k, ce, _flops| {
            println!("  eval step={step} K={k} CE={ce:.4}");
        }),
        Some(&mut |old, new, step, _flops| {
            println!("  GROW K={old}->{new} at step {step}");
        }),
    );

    println!(
        "done: best={:.4} final={:.4} steps={} elapsed={:.1}s",
        result.best_loss, result.final_loss, result.total_steps, result.elapsed
    );
    let _ = model; // keep model alive
}
