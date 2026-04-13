use burn::prelude::*;
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;

// ── ActiveLayerNorm ──────────────────────────────────────────
// LayerNorm restricted to first d_active dimensions.
// Required for width elasticity: standard LN breaks preservation
// because zero-padded dimensions change the statistics.

#[derive(Module, Debug)]
pub struct ActiveLayerNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub bias: Param<Tensor<B, 1>>,
    #[module(skip)]
    pub d_active: usize,
    #[module(skip)]
    pub eps: f64,
}

impl<B: Backend> ActiveLayerNorm<B> {
    pub fn new(d: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([d], device)),
            bias: Param::from_tensor(Tensor::zeros([d], device)),
            d_active: d,
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, d_total] = x.dims();
        let da = self.d_active;

        // Extract active dimensions
        let active = x.clone().slice([0..b, 0..t, 0..da]); // [b, t, da]

        // Compute stats over active dims only
        let mean = active.clone().mean_dim(2); // [b, t, 1]
        let centered = active - mean.clone();
        let var = centered.clone().powf_scalar(2.0).mean_dim(2); // [b, t, 1]
        let normed = centered / (var + self.eps).sqrt();

        // Apply weight and bias (active dims only)
        let w = self.weight.val().slice([0..da]).unsqueeze::<2>().unsqueeze::<3>(); // [1, 1, da]
        let bias = self.bias.val().slice([0..da]).unsqueeze::<2>().unsqueeze::<3>();
        let scaled = normed * w + bias;

        // Output: active dims normalized, rest zero
        let out = Tensor::zeros([b, t, d_total], &x.device());
        out.slice_assign([0..b, 0..t, 0..da], scaled)
    }

    /// Grow capacity to new_d: zero-pad weight/bias.
    /// d_active is NOT changed — call set_d_active() to activate new dims.
    pub fn grow(&mut self, new_d: usize, device: &B::Device) {
        let old_d = self.weight.val().dims()[0];
        if new_d <= old_d {
            return;
        }
        let old_w = self.weight.val();
        let old_b = self.bias.val();
        let new_w = Tensor::ones([new_d], device)
            .slice_assign([0..old_d], old_w);
        let new_b = Tensor::zeros([new_d], device)
            .slice_assign([0..old_d], old_b);
        self.weight = Param::from_tensor(new_w);
        self.bias = Param::from_tensor(new_b);
    }

    /// Activate dimensions up to d.
    pub fn set_d_active(&mut self, d: usize) {
        self.d_active = d;
    }
}

// ── Snowman model ──────────────────────────────────────────

#[derive(Module, Debug)]
pub struct Snowman<B: Backend> {
    pub embed_tok: Embedding<B>,
    pub pe: Param<Tensor<B, 3>>,
    pub block_ln: ActiveLayerNorm<B>,
    pub block_ff1: Linear<B>,
    pub block_ff2: Linear<B>,
    pub readout_ln: ActiveLayerNorm<B>,
    pub gates: Param<Tensor<B, 1>>,
    #[module(skip)]
    pub k_max: usize,
    #[module(skip)]
    pub vocab: usize,
    #[module(skip)]
    pub d: usize,
    #[module(skip)]
    pub d_ff: usize,
    #[module(skip)]
    pub seq_len: usize,
}

#[derive(Config, Debug)]
pub struct SnowmanConfig {
    pub vocab: usize,
    pub d: usize,
    pub d_ff: usize,
    pub seq_len: usize,
    pub k_max: usize,
}

impl SnowmanConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Snowman<B> {
        let mut gate_data = vec![0.0f32; self.k_max];
        gate_data[0] = 1.0;
        let gates = Param::from_tensor(
            Tensor::from_floats(gate_data.as_slice(), device),
        );

        Snowman {
            embed_tok: EmbeddingConfig::new(self.vocab, self.d).init(device),
            pe: Param::from_tensor(sinusoidal_pe::<B>(self.seq_len, self.d, device)),
            block_ln: ActiveLayerNorm::new(self.d, device),
            block_ff1: LinearConfig::new(self.d, self.d_ff).init(device),
            block_ff2: LinearConfig::new(self.d_ff, self.d).init(device),
            readout_ln: ActiveLayerNorm::new(self.d, device),
            gates,
            k_max: self.k_max,
            vocab: self.vocab,
            d: self.d,
            d_ff: self.d_ff,
            seq_len: self.seq_len,
        }
    }
}

/// Fixed sinusoidal position encoding (not learnable).
/// Returns tensor of shape [1, seq_len, d].
fn sinusoidal_pe<B: Backend>(seq_len: usize, d: usize, device: &B::Device) -> Tensor<B, 3> {
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

impl<B: Backend> Snowman<B> {
    /// Embed: tok(x) + stored sinusoidal PE (sliced to seq length)
    fn embed(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_b, t] = x.dims();
        let pe = self.pe.val().slice([0..1, 0..t, 0..self.d]);
        self.embed_tok.forward(x) + pe
    }

    /// Block: h + ff2(gelu(ff1(ln(h))))
    fn block(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.block_ln.forward(h.clone());
        let ff = self.block_ff1.forward(normed);
        let ff = gelu(ff);
        let ff = self.block_ff2.forward(ff);
        h + ff
    }

    /// Readout: ln(h) @ embed_tok.weight^T  (weight tying)
    fn readout(&self, h: Tensor<B, 3>) -> Tensor<B, 3> {
        let normed = self.readout_ln.forward(h); // [b, t, d]
        let weight = self.embed_tok.weight.val(); // [vocab, d]
        let weight_3d = weight.unsqueeze::<3>();  // [1, vocab, d]
        normed.matmul(weight_3d.transpose())      // [b, t, vocab]
    }

    /// Full forward for inference.
    pub fn forward(&self, x: Tensor<B, 2, Int>, k: usize) -> Tensor<B, 3> {
        let mut h = self.embed(x);
        let gates = self.gates.val();
        for i in 0..k {
            let h_new = self.block(h.clone());
            let alpha = gates.clone().slice([i..i + 1])
                .unsqueeze::<2>().unsqueeze::<3>(); // [1,1,1]
            h = h.clone() + (h_new - h) * alpha;
        }
        self.readout(h)
    }

    /// Single-depth forward+backward for training at depth k.
    /// Detaches between depths: O(1) activation memory.
    pub fn forward_single(
        &self,
        x: Tensor<B, 2, Int>,
        k: usize,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1>
    where
        B: burn::tensor::backend::AutodiffBackend,
    {
        let mut h = self.embed(x);
        let gates = self.gates.val();
        // Intermediate depths: detach to avoid graph construction
        for i in 0..k.saturating_sub(1) {
            h = h.detach();
            let h_new = self.block(h.clone());
            let alpha = gates.clone().slice([i..i + 1])
                .unsqueeze::<2>().unsqueeze::<3>();
            h = h.clone() + (h_new - h) * alpha;
        }
        // Final depth: full gradient tracking
        h = h.detach();
        let h_new = self.block(h.clone());
        let alpha = gates.clone().slice([k - 1..k])
            .unsqueeze::<2>().unsqueeze::<3>();
        h = h.clone() + (h_new - h) * alpha;

        let logits = self.readout(h); // [b, t, vocab]
        let [b, t, v] = logits.dims();
        let logits_flat = logits.reshape([b * t, v]);
        let targets_flat = targets.reshape([b * t]);
        burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat)
    }

    pub fn gate_values(&self) -> Vec<f32> {
        let g = self.gates.val();
        let data = g.to_data();
        data.to_vec::<f32>().unwrap()
    }

    pub fn embed_param_count(&self) -> usize {
        self.vocab * self.d
    }

    pub fn block_param_count(&self) -> usize {
        self.d * self.d_ff + self.d_ff + self.d_ff * self.d + self.d + 2 * self.d
    }

    /// Readout FLOP cost: LayerNorm + tied V×d matmul.
    pub fn readout_param_count(&self) -> usize {
        2 * self.d + self.vocab * self.d
    }

    pub fn param_count(&self) -> usize {
        // PE is stored but not learnable (sinusoidal, fixed). Not counted in trainable params.
        self.embed_param_count() + self.block_param_count() + 2 * self.d + self.k_max
    }

    /// Grow width: zero-pad all layers to new_d.
    /// Preserves existing weights exactly. New dimensions initialized to zero.
    pub fn grow_width(mut self, new_d: usize, new_d_ff: usize, device: &B::Device) -> Self {
        let old_d = self.d;
        let old_d_ff = self.d_ff;
        if new_d == old_d {
            return self;
        }

        // Embedding: zero-pad [vocab, old_d] -> [vocab, new_d]
        let old_emb = self.embed_tok.weight.val(); // [vocab, old_d]
        let new_emb = Tensor::zeros([self.vocab, new_d], device)
            .slice_assign([0..self.vocab, 0..old_d], old_emb);
        self.embed_tok = EmbeddingConfig::new(self.vocab, new_d).init(device);
        self.embed_tok.weight = Param::from_tensor(new_emb);

        // PE: zero-pad [1, seq_len, old_d] -> [1, seq_len, new_d]
        let old_pe = self.pe.val(); // [1, seq_len, old_d]
        let new_pe = Tensor::zeros([1, self.seq_len, new_d], device)
            .slice_assign([0..1, 0..self.seq_len, 0..old_d], old_pe);
        self.pe = Param::from_tensor(new_pe);

        // ActiveLayerNorm: grow
        self.block_ln.grow(new_d, device);
        self.readout_ln.grow(new_d, device);

        // FF1: Linear(d, d_ff). Burn weight shape = [d, d_ff] (in, out).
        // Zero-pad: [old_d, old_d_ff] -> [new_d, new_d_ff]
        let old_ff1_w = self.block_ff1.weight.val();
        let new_ff1_w = Tensor::zeros([new_d, new_d_ff], device)
            .slice_assign([0..old_d, 0..old_d_ff], old_ff1_w);
        let new_ff1_b = if let Some(old_b) = self.block_ff1.bias.as_ref() {
            let ob = old_b.val();
            Some(Param::from_tensor(
                Tensor::zeros([new_d_ff], device)
                    .slice_assign([0..old_d_ff], ob),
            ))
        } else {
            None
        };
        self.block_ff1 = LinearConfig::new(new_d, new_d_ff).init(device);
        self.block_ff1.weight = Param::from_tensor(new_ff1_w);
        if let Some(b) = new_ff1_b {
            self.block_ff1.bias = Some(b);
        }

        // FF2: Linear(d_ff, d). Burn weight shape = [d_ff, d] (in, out).
        // Zero-pad: [old_d_ff, old_d] -> [new_d_ff, new_d]
        let old_ff2_w = self.block_ff2.weight.val();
        let new_ff2_w = Tensor::zeros([new_d_ff, new_d], device)
            .slice_assign([0..old_d_ff, 0..old_d], old_ff2_w);
        let new_ff2_b = if let Some(old_b) = self.block_ff2.bias.as_ref() {
            let ob = old_b.val();
            Some(Param::from_tensor(
                Tensor::zeros([new_d], device)
                    .slice_assign([0..old_d], ob),
            ))
        } else {
            None
        };
        self.block_ff2 = LinearConfig::new(new_d_ff, new_d).init(device);
        self.block_ff2.weight = Param::from_tensor(new_ff2_w);
        if let Some(b) = new_ff2_b {
            self.block_ff2.bias = Some(b);
        }

        self.d = new_d;
        self.d_ff = new_d_ff;
        // NOTE: ALN d_active is NOT changed. Call activate_width() to use new dims.
        self
    }

    /// Activate new width dimensions in ALN layers.
    /// Call this after grow_width() when ready to train at new width.
    pub fn activate_width(&mut self) {
        self.block_ln.set_d_active(self.d);
        self.readout_ln.set_d_active(self.d);
    }

    /// Grow or shrink gate list. New gates init to 0 (identity).
    pub fn resize_depth(mut self, new_k_max: usize, device: &B::Device) -> Self {
        let old = self.k_max;
        if new_k_max == old {
            return self;
        }
        let old_gates = self.gates.val();
        let mut new_data = vec![0.0f32; new_k_max];
        let copy_len = old.min(new_k_max);
        let old_data: Vec<f32> = old_gates.to_data().to_vec().unwrap();
        new_data[..copy_len].copy_from_slice(&old_data[..copy_len]);
        self.gates = Param::from_tensor(Tensor::from_floats(new_data.as_slice(), device));
        self.k_max = new_k_max;
        self
    }
}
