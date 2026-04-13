use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::prelude::*;
use burn::nn::{
    EmbeddingConfig, Linear, LinearConfig, LayerNorm, LayerNormConfig,
    loss::CrossEntropyLossConfig,
};
use burn::tensor::activation::gelu;

use ski_creche::model::SnowmanConfig;

type B = Autodiff<Wgpu>;

fn vram_estimate(
    params: usize,
    n_act_layers: usize,
    batch: usize,
    seq_len: usize,
    d: usize,
    d_ff: usize,
) -> usize {
    let param_bytes = params * 4;
    let optim_bytes = params * 4 * 2;
    let grad_bytes = params * 4;
    let act_per_layer = batch * seq_len * (d + d_ff) * 4;
    let act_bytes = n_act_layers * act_per_layer;
    param_bytes + optim_bytes + grad_bytes + act_bytes
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Fixed sinusoidal position encoding.
fn sinusoidal_pe(seq_len: usize, d: usize, device: &<B as Backend>::Device) -> Tensor<B, 3> {
    let mut pe = vec![0.0f32; seq_len * d];
    for pos in 0..seq_len {
        for i in (0..d).step_by(2) {
            let div = (10000.0f32).powf(i as f32 / d as f32);
            pe[pos * d + i] = (pos as f32 / div).sin();
            if i + 1 < d {
                pe[pos * d + i + 1] = (pos as f32 / div).cos();
            }
        }
    }
    Tensor::<B, 1>::from_floats(pe.as_slice(), device).reshape([1, seq_len, d])
}

fn e2e_forward_loss(
    vocab: usize, d: usize, d_ff: usize, seq_len: usize, batch: usize, k: usize,
    device: &<B as Backend>::Device,
) -> f64 {
    let tok = EmbeddingConfig::new(vocab, d).init::<B>(device);
    let ln_f = LayerNormConfig::new(d).init::<B>(device);

    let block_lns: Vec<LayerNorm<B>> = (0..k).map(|_| LayerNormConfig::new(d).init(device)).collect();
    let block_ff1s: Vec<Linear<B>> = (0..k).map(|_| LinearConfig::new(d, d_ff).init(device)).collect();
    let block_ff2s: Vec<Linear<B>> = (0..k).map(|_| LinearConfig::new(d_ff, d).init(device)).collect();

    let data = Tensor::<B, 2, Int>::random(
        [batch, seq_len + 1],
        burn::tensor::Distribution::Uniform(0.0, vocab as f64),
        device,
    );
    let x = data.clone().slice([0..batch, 0..seq_len]);
    let y = data.slice([0..batch, 1..seq_len + 1]);

    let pe = sinusoidal_pe(seq_len, d, device);
    let mut h = tok.forward(x) + pe;

    for i in 0..k {
        let normed = block_lns[i].forward(h.clone());
        h = h + block_ff2s[i].forward(gelu(block_ff1s[i].forward(normed)));
    }

    let embed_w = tok.weight.val().unsqueeze::<3>(); // [1, vocab, d]
    let logits = ln_f.forward(h).matmul(embed_w.transpose()); // [b, t, vocab]
    let [b, t, v] = logits.dims();
    let loss = CrossEntropyLossConfig::new()
        .init(device)
        .forward(logits.reshape([b * t, v]), y.reshape([b * t]));
    loss.into_scalar().elem::<f64>()
}

fn main() {
    let device = WgpuDevice::default();

    let vocab = 256usize;
    let d = 384usize;
    let d_ff = 1536usize;
    let seq_len = 256usize;
    let batch = 32usize;

    // Smaller dims for actual forward/backward
    let sd = 64usize;
    let sd_ff = 256usize;
    let sseq = 32usize;
    let sbatch = 16usize;

    println!("Config: d={d}, d_ff={d_ff}, seq={seq_len}, batch={batch}, vocab={vocab}");
    println!("Backend: burn-wgpu (GPU)");
    println!("(Forward/backward with d={sd} for speed)\n");

    println!("{:>3} | {:>12} {:>10} {:>10} | {:>12} {:>10} {:>10} | {:>6}",
        "K", "E2E params", "E2E loss", "E2E VRAM",
        "Snow params", "Snow loss", "Snow VRAM", "ratio");
    println!("{}", "-".repeat(95));

    for &k in &[2usize, 4, 8] {
        // E2E params at full scale (no pos embedding params — sinusoidal is free)
        let e2e_params = {
            let embed_p = vocab * d;
            let block_p = d * d_ff + d_ff + d_ff * d + d + 2 * d;
            embed_p + k * block_p + 2 * d
        };

        let e2e_loss = e2e_forward_loss(vocab, sd, sd_ff, sseq, sbatch, k, &device);

        // Snowball
        let snow_cfg = SnowmanConfig::new(vocab, sd, sd_ff, sseq, k);
        let snow_model = snow_cfg.init::<B>(&device);

        let snow_loss = {
            let data = Tensor::<B, 2, Int>::random(
                [sbatch, sseq + 1],
                burn::tensor::Distribution::Uniform(0.0, vocab as f64),
                &device,
            );
            let x = data.clone().slice([0..sbatch, 0..sseq]);
            let y = data.slice([0..sbatch, 1..sseq + 1]);
            let loss = snow_model.forward_single(x, k, y);
            loss.into_scalar().elem::<f64>()
        };

        // Theoretical VRAM at full scale
        let e2e_vram = vram_estimate(e2e_params, k, batch, seq_len, d, d_ff);
        let snow_full_params = {
            let embed_p = vocab * d;
            let block_p = d * d_ff + d_ff + d_ff * d + d + 2 * d;
            embed_p + block_p + 2 * d + k
        };
        let snow_vram = vram_estimate(snow_full_params, 1, batch, seq_len, d, d_ff);
        let ratio = snow_vram as f64 / e2e_vram as f64;

        println!("{:>3} | {:>12} {:>10.4} {:>8.0} MB | {:>12} {:>10.4} {:>8.0} MB | {:>5.2}x",
            k, e2e_params, e2e_loss, mb(e2e_vram),
            snow_full_params, snow_loss, mb(snow_vram), ratio);
    }

    println!();
    println!("Key observations:");
    println!("  E2E params grow as O(K) — each depth has its own block weights.");
    println!("  Snowball params are O(1) — one shared block, reused K times.");
    println!("  E2E activations grow as O(K) — all layers held for backward.");
    println!("  Snowball activations are O(1) — detach between depths, one depth at a time.");
}
