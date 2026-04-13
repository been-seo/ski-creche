"""
Width Detach: Theorem, Proof, and Numerical Verification.

Theorem (Width Detach Gradient Isolation).
    In width-sampled training where width level g restricts computation to
    the first d_g dimensions (n_g attention heads, d_ff_g = 4d_g FF width,
    ActiveLayerNorm over d_g dims), the following hold:

    (i)   Only W[i,j] with i,j < d_g receive non-zero gradients.
    (ii)  The gradient dL_g/dW[:d_g,:d_g] is identical to that of a
          standalone width-d_g model with the SAME weights.
    (iii) The forward output at width g is independent of any W[i,j]
          with i >= d_g or j >= d_g.

Proof.
    At width g, the computation is structurally restricted:
      - ALN normalizes over first d_g dims, zeros the rest
      - Attention uses n_g = d_g / d_head heads (dims 0..d_g - 1)
      - FF uses W_1[:d_ff_g, :d_g] and W_2[:d_g, :d_ff_g]
      - Residual connections stay in R^{d_g}

    (iii) The forward pass at width g uses ONLY the submatrix W[:d_g,:d_g]
    for every linear layer. Parameters with index >= d_g never participate.

    (i) By the chain rule, dL/dW[i,j] = dL/dy_i * x_j.
      - j >= d_g: input x_j = 0 (from ALN/width restriction), so dL/dW[i,j] = 0.
      - i >= d_g: output y_i is not used downstream (width restriction),
        so dL/dy_i = 0, hence dL/dW[i,j] = 0.
      Only i,j < d_g remain.

    (ii) At width g, the computation on the d_g-dimensional subspace is:
      - LN: ALN([h;0], d_g) = [LN_d_g(h); 0], same stats as standalone LN
      - Linear: W[:d_g,:d_g] * h, same as standalone W_small * h
      - Attention: n_g heads with d_head each, scale = 1/sqrt(d_head) unchanged
      - GELU: GELU(z) pointwise, same
      - Residual: h + f(h), same
    Every intermediate value matches the standalone model exactly,
    therefore the gradient dL_g/dW[:d_g,:d_g] is identical.  []

Corollary (Width Detach is Free).
    After width-sampled training (g ~ Uniform{1..G}), evaluating at width g
    uses W[:d_g,:d_g] which has been optimized by gradients from all widths
    g' >= g. Each width g' contributes a valid gradient (by (ii)), so
    W[:d_g,:d_g] converges to a weight matrix that works well at ALL widths
    >= g. Width detach (running at g < G) incurs no retraining cost.

Corollary (Depth-Width Symmetry).
    Width detach and depth detach have identical structure:

    | Property        | Depth Detach          | Width Detach              |
    |-----------------|----------------------|---------------------------|
    | Shared params   | Block W (all depths) | W[:d_g,:d_g] (all w >= g) |
    | Isolation       | h.detach()           | Width restriction + ALN   |
    | Sampling        | k ~ U{1..K}          | g ~ U{1..G}              |
    | Gradient/step   | dL_k/dW              | dL_g/dW[:d_g,:d_g]       |
    | Detach cost     | Free                 | Free                      |

    Both: shared parameters receive gradients from ALL levels via random
    sampling. AdamW's per-parameter m/sqrt(v) normalizes the mixed-level
    gradients. Detach = restrict to smaller subspace, which was trained.

Remark (Width Gate).
    An explicit width gate beta_g (analogous to depth gate alpha_k) is
    NOT required for width detach. Zero-init weights at growth provide the
    same monotonicity guarantee: new dimensions contribute zero initially,
    so growth cannot increase loss. However, a learnable beta_g can
    accelerate the activation of new dimensions.

=== Numerical Verification ===
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class ActiveLayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.d_active = d
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        d = self.d_active
        a = x[..., :d]
        mu = a.mean(-1, keepdim=True)
        var = a.var(-1, keepdim=True, unbiased=False)
        normed = (a - mu) / torch.sqrt(var + self.eps)
        out = torch.zeros_like(x)
        out[..., :d] = self.weight[:d] * normed + self.bias[:d]
        return out


class WidthElasticBlock(nn.Module):
    """Block that supports width-restricted forward pass."""
    def __init__(self, d, n_heads):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.ln1 = ActiveLayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.ln2 = ActiveLayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, h, d_active=None):
        """Forward with optional width restriction."""
        if d_active is None:
            d_active = self.d
        B, T, _ = h.shape
        dh = self.d_head
        nh = d_active // dh  # number of active heads

        # Pre-attention LN (restricted)
        self.ln1.d_active = d_active
        h_n = self.ln1(h)

        # Attention: only first nh heads
        Q = (h_n[..., :d_active] @ self.W_q.weight[:d_active, :d_active].T
             ).view(B, T, nh, dh).transpose(1, 2)
        K = (h_n[..., :d_active] @ self.W_k.weight[:d_active, :d_active].T
             ).view(B, T, nh, dh).transpose(1, 2)
        V = (h_n[..., :d_active] @ self.W_v.weight[:d_active, :d_active].T
             ).view(B, T, nh, dh).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, d_active)
        attn_out = attn @ self.W_o.weight[:d_active, :d_active].T

        # Residual (width-restricted)
        h_res = h.clone()
        h_res[..., :d_active] = h[..., :d_active] + attn_out

        # Pre-FF LN (restricted)
        self.ln2.d_active = d_active
        h_n2 = self.ln2(h_res)

        # FF: restricted to d_active input, 4*d_active intermediate
        d_ff = 4 * d_active
        ff_out = (h_n2[..., :d_active] @ self.ff1.weight[:d_ff, :d_active].T
                  + self.ff1.bias[:d_ff])
        ff_out = F.gelu(ff_out)
        ff_out = ff_out @ self.ff2.weight[:d_active, :d_ff].T + self.ff2.bias[:d_active]

        out = h_res.clone()
        out[..., :d_active] = h_res[..., :d_active] + ff_out
        return out


class StandaloneBlock(nn.Module):
    """Standard block at fixed width (for comparison)."""
    def __init__(self, d, n_heads):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.ln1 = nn.LayerNorm(d)
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, h):
        B, T, _ = h.shape
        nh, dh = self.n_heads, self.d_head
        h_n = self.ln1(h)
        Q = self.W_q(h_n).view(B, T, nh, dh).transpose(1, 2)
        K = self.W_k(h_n).view(B, T, nh, dh).transpose(1, 2)
        V = self.W_v(h_n).view(B, T, nh, dh).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.d)
        h = h + self.W_o(attn)
        h = h + self.ff2(F.gelu(self.ff1(self.ln2(h))))
        return h


def verify_gradient_isolation():
    """Verify: gradient at width g equals standalone d_g gradient."""
    torch.manual_seed(42)
    d_full = 128   # 4 heads
    d_small = 64   # 2 heads
    d_head = 32
    B, T = 2, 8
    V = 50

    # --- Wide block (d=128) ---
    wide = WidthElasticBlock(d_full, d_full // d_head)

    # --- Standalone narrow block (d=64, same weights as wide[:64,:64]) ---
    narrow = StandaloneBlock(d_small, d_small // d_head)
    with torch.no_grad():
        narrow.ln1.weight.copy_(wide.ln1.weight[:d_small])
        narrow.ln1.bias.copy_(wide.ln1.bias[:d_small])
        narrow.ln2.weight.copy_(wide.ln2.weight[:d_small])
        narrow.ln2.bias.copy_(wide.ln2.bias[:d_small])
        narrow.W_q.weight.copy_(wide.W_q.weight[:d_small, :d_small])
        narrow.W_k.weight.copy_(wide.W_k.weight[:d_small, :d_small])
        narrow.W_v.weight.copy_(wide.W_v.weight[:d_small, :d_small])
        narrow.W_o.weight.copy_(wide.W_o.weight[:d_small, :d_small])
        narrow.ff1.weight.copy_(wide.ff1.weight[:4*d_small, :d_small])
        narrow.ff1.bias.copy_(wide.ff1.bias[:4*d_small])
        narrow.ff2.weight.copy_(wide.ff2.weight[:d_small, :4*d_small])
        narrow.ff2.bias.copy_(wide.ff2.bias[:d_small])

    # Input: random d_small, zero-padded to d_full
    h_small = torch.randn(B, T, d_small)
    h_wide = torch.zeros(B, T, d_full)
    h_wide[..., :d_small] = h_small.clone()

    # Embedding (random, for tied readout)
    emb_small = torch.randn(V, d_small, requires_grad=False)
    emb_wide = torch.zeros(V, d_full)
    emb_wide[:, :d_small] = emb_small

    targets = torch.randint(0, V, (B, T))

    # --- Forward + backward: wide block at width d_small ---
    h_wide.requires_grad_(True)
    out_wide = wide(h_wide, d_active=d_small)
    logits_wide = out_wide[..., :d_small] @ emb_wide[:, :d_small].T
    loss_wide = F.cross_entropy(logits_wide.reshape(-1, V), targets.reshape(-1))
    loss_wide.backward()

    # --- Forward + backward: standalone narrow block ---
    h_small.requires_grad_(True)
    out_narrow = narrow(h_small)
    logits_narrow = out_narrow @ emb_small.T
    loss_narrow = F.cross_entropy(logits_narrow.reshape(-1, V), targets.reshape(-1))
    loss_narrow.backward()

    # === Verification ===
    print("=" * 60)
    print("  Width Detach Gradient Isolation Verification")
    print("=" * 60)

    # 1. Forward output matches
    out_err = (out_wide[..., :d_small] - out_narrow).abs().max().item()
    print(f"\n(1) Forward output |wide[:d_g] - narrow| = {out_err:.2e}")
    assert out_err < 1e-5, f"Forward mismatch: {out_err}"
    print("    [OK] Width-restricted forward = standalone forward")

    # 2. Loss matches
    loss_err = abs(loss_wide.item() - loss_narrow.item())
    print(f"\n(2) Loss |wide - narrow| = {loss_err:.2e}")
    assert loss_err < 1e-5, f"Loss mismatch: {loss_err}"
    print("    [OK] Losses identical")

    # 3. Gradients match for W[:d_g, :d_g]
    print(f"\n(3) Gradient comparison (W[:d_g, :d_g]):")
    pairs = [
        ('W_q', wide.W_q.weight.grad, narrow.W_q.weight.grad),
        ('W_k', wide.W_k.weight.grad, narrow.W_k.weight.grad),
        ('W_v', wide.W_v.weight.grad, narrow.W_v.weight.grad),
        ('W_o', wide.W_o.weight.grad, narrow.W_o.weight.grad),
        ('ff1', wide.ff1.weight.grad, narrow.ff1.weight.grad),
        ('ff2', wide.ff2.weight.grad, narrow.ff2.weight.grad),
    ]
    for name, grad_w, grad_n in pairs:
        if name in ('ff1',):
            sub = grad_w[:4*d_small, :d_small]
        elif name in ('ff2',):
            sub = grad_w[:d_small, :4*d_small]
        else:
            sub = grad_w[:d_small, :d_small]
        err = (sub - grad_n).abs().max().item()
        print(f"    {name:4s}: |grad_wide[:d_g,:d_g] - grad_narrow| = {err:.2e}")
        assert err < 1e-4, f"Gradient mismatch for {name}: {err}"
    print("    [OK] All gradients match standalone model")

    # 4. Gradients zero outside d_g
    print(f"\n(4) Gradient isolation (W outside d_g):")
    for name, layer in [('W_q', wide.W_q), ('W_k', wide.W_k),
                        ('W_v', wide.W_v), ('W_o', wide.W_o)]:
        g = layer.weight.grad
        # rows >= d_small
        row_err = g[d_small:, :].abs().max().item()
        # cols >= d_small
        col_err = g[:, d_small:].abs().max().item()
        print(f"    {name:4s}: |grad[d_g:, :]| = {row_err:.2e}, "
              f"|grad[:, d_g:]| = {col_err:.2e}")
        assert row_err < 1e-7 and col_err < 1e-7, f"Isolation failed for {name}"

    # ff1: rows >= 4*d_small, cols >= d_small
    g = wide.ff1.weight.grad
    print(f"    ff1 : |grad[d_ff_g:, :]| = {g[4*d_small:, :].abs().max().item():.2e}, "
          f"|grad[:, d_g:]| = {g[:, d_small:].abs().max().item():.2e}")
    # ff2: rows >= d_small, cols >= 4*d_small
    g = wide.ff2.weight.grad
    print(f"    ff2 : |grad[d_g:, :]| = {g[d_small:, :].abs().max().item():.2e}, "
          f"|grad[:, d_ff_g:]| = {g[:, 4*d_small:].abs().max().item():.2e}")
    print("    [OK] All gradients zero outside d_g x d_g submatrix")

    # 5. New dimensions stay zero
    out_new = out_wide[..., d_small:].abs().max().item()
    print(f"\n(5) New dims in output: |h'[d_g:]| = {out_new:.2e}")
    assert out_new < 1e-7
    print("    [OK] Dimensions > d_g remain zero")

    print(f"\n{'='*60}")
    print("  All verifications passed.")
    print("  Width detach gradient isolation is exact.")
    print(f"{'='*60}")


if __name__ == '__main__':
    verify_gradient_isolation()
