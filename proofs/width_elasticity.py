"""
Width Elasticity: Theorem, Proof, and Numerical Verification.

Theorem (Exact Width Preservation).
    Let M_d be a Snowman model with hidden dimension d, n_h attention heads
    (d_head = d / n_h). Define M_{d'} with d' = d + n_new · d_head by:
      (i)    Embedding:  E' = [E | 0] ∈ R^{V × d'}
      (ii)   Attention:  add n_new heads with W_Q, W_K, W_V, W_O = 0
      (iii)  FF:         zero-pad  W_1 ∈ R^{d_ff × d} → R^{d'_ff × d'},
                                   W_2 ∈ R^{d × d_ff} → R^{d' × d'_ff}
      (iv)   LayerNorm → ActiveLayerNorm with active set A = {0, ..., d-1}
      (v)    Readout:   tied with E' (automatic)
    Then M_{d'}(x, K) = M_d(x, K) for all inputs x and all depths K.

Proof.
    Lemma: if input to any layer is [h; 0] (h ∈ R^d, padded to R^{d'}),
    the output is [f(h); 0] where f is the original layer.

    (1) ActiveLayerNorm.
        ALN([h;0], A={0..d-1}): stats computed over first d dims only.
        μ_A = mean(h), σ_A = std(h).
        Output: [LN(h); 0].  [OK]

    (2) Multi-head attention with new zero-init heads.
        After ALN, input is [LN(h); 0].

        Old heads (j ≤ n_h):
          Q_j = [LN(h); 0] · W'_{Q,j}
          W'_{Q,j} ∈ R^{d' × d_head}, first d rows = W_{Q,j}, rest = 0.
          → Q_j = LN(h) · W_{Q,j}  (zero rows don't contribute).
          Same for K_j, V_j. Identical attention weights, identical output.

        New heads (j > n_h):
          W'_{V,j} = 0 → V_j = 0 → head_j = softmax(·) · 0 = 0.
          (Scaling: 1/√d_head is unchanged because d_head is unchanged.)

        Output projection W'_O ∈ R^{d' × d'}:
          W'_O = [[W_O, 0], [0, 0]].
          Input = [concat_old; 0].
          Output = [W_O · concat_old; 0] = [Attn(LN(h)); 0].  [OK]

    (3) Feed-forward.
        W'_1 = [[W_1, 0], [0, 0]], b'_1 = [b_1; 0].
        Input [h; 0]: W'_1 [h;0] + b'_1 = [W_1 h + b_1; 0].
        GELU([z; 0]) = [GELU(z); 0]  since GELU(0) = 0.
        W'_2 [GELU(z); 0] + b'_2 = [FF(h); 0].  [OK]

    (4) Residual.  [h;0] + [f(h);0] = [h+f(h); 0].  [OK]

    (5) Gated residual (depth gate α_k).
        [h;0] + α_k · ([Block(h);0] - [h;0]) = [h + α_k·(Block(h)-h); 0].  [OK]

    (6) Readout (tied).
        logits = E'^T [h;0] = [E|0]^T [h;0] = E^T h.
        Identical to M_d.  [OK]

    By induction on layers, M_{d'}(x,K) = M_d(x,K).  □

Corollary (Monotonic Width Growth).
    Define width gate β_g ∈ R (init 0) per dimension group g.
    When β_g = 0, group g contributes zero → loss unchanged.
    Combined with depth gate α_k = 0, the model is 2D-elastic:
    neither adding depth nor width can increase loss.

Remark (Width Detach).
    Width detach is NOT free, unlike depth detach.
    After training at d', gradients flow through all d' dimensions,
    modifying the first d dimensions. Shrinking to d gives a model
    different from the original d-width model.
    (Depth detach is free because per-depth training with h.detach()
    makes each depth independent. No analogous mechanism exists for width
    without width-sampled training à la slimmable networks.)

Remark (Attention Scaling).
    Width must grow by adding NEW heads, not widening existing ones.
    Widening changes d_head, which changes the attention scaling 1/√d_head,
    breaking exact preservation. Adding heads keeps d_head constant.
    This constrains d to grow in multiples of d_head.

=== Numerical Verification ===
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ActiveLayerNorm(nn.Module):
    """LayerNorm restricted to active dimensions."""
    def __init__(self, d_active: int, d_total: int, eps: float = 1e-5):
        super().__init__()
        self.d_active = d_active
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_total))
        self.bias = nn.Parameter(torch.zeros(d_total))

    def forward(self, x):
        d = self.d_active
        active = x[..., :d]
        mu = active.mean(dim=-1, keepdim=True)
        var = active.var(dim=-1, keepdim=True, unbiased=False)
        normed = (active - mu) / torch.sqrt(var + self.eps)
        out = torch.zeros_like(x)
        out[..., :d] = self.weight[:d] * normed + self.bias[:d]
        return out


def verify():
    torch.manual_seed(42)
    d = 64
    n_heads = 4
    d_head = d // n_heads  # 16
    n_new = 2  # add 2 heads
    d_prime = d + n_new * d_head  # 64 + 32 = 96
    d_ff = 4 * d
    d_ff_prime = 4 * d_prime
    V, T, B = 100, 16, 4

    # ========================================
    # Original model (d=64, 4 heads)
    # ========================================
    emb = nn.Embedding(V, d)
    ln1 = nn.LayerNorm(d)
    ln2 = nn.LayerNorm(d)
    # Attention: separate Q,K,V,O projections
    W_q = nn.Linear(d, d, bias=False)
    W_k = nn.Linear(d, d, bias=False)
    W_v = nn.Linear(d, d, bias=False)
    W_o = nn.Linear(d, d, bias=False)
    # FF
    ff1 = nn.Linear(d, d_ff)
    ff2 = nn.Linear(d_ff, d)

    x = torch.randint(0, V, (B, T))

    def forward_orig(x):
        h = emb(x)
        # Attention block
        h_n = ln1(h)
        Q = W_q(h_n).view(B, T, n_heads, d_head).transpose(1, 2)
        K = W_k(h_n).view(B, T, n_heads, d_head).transpose(1, 2)
        Vt = W_v(h_n).view(B, T, n_heads, d_head).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, Vt, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d)
        h = h + W_o(attn_out)
        # FF block
        h = h + ff2(F.gelu(ff1(ln2(h))))
        return F.linear(h, emb.weight)

    logits_orig = forward_orig(x)

    # ========================================
    # Widened model (d'=96, 6 heads)
    # ========================================
    n_heads_w = n_heads + n_new  # 6

    # Embedding: zero-pad
    emb_w = nn.Embedding(V, d_prime)
    with torch.no_grad():
        emb_w.weight.zero_()
        emb_w.weight[:, :d] = emb.weight

    # ActiveLayerNorm (active = first d dims)
    aln1 = ActiveLayerNorm(d_active=d, d_total=d_prime)
    aln2 = ActiveLayerNorm(d_active=d, d_total=d_prime)
    with torch.no_grad():
        aln1.weight[:d] = ln1.weight; aln1.bias[:d] = ln1.bias
        aln2.weight[:d] = ln2.weight; aln2.bias[:d] = ln2.bias

    # Attention: zero-pad Q,K,V,O
    # W_q: R^{d×d} → R^{d'×d'}, with old heads preserved, new heads = 0
    W_q_w = nn.Linear(d_prime, d_prime, bias=False)
    W_k_w = nn.Linear(d_prime, d_prime, bias=False)
    W_v_w = nn.Linear(d_prime, d_prime, bias=False)
    W_o_w = nn.Linear(d_prime, d_prime, bias=False)
    with torch.no_grad():
        for Ww, Wo in [(W_q_w, W_q), (W_k_w, W_k), (W_v_w, W_v), (W_o_w, W_o)]:
            Ww.weight.zero_()
            Ww.weight[:d, :d] = Wo.weight

    # FF: zero-pad
    ff1_w = nn.Linear(d_prime, d_ff_prime)
    ff2_w = nn.Linear(d_ff_prime, d_prime)
    with torch.no_grad():
        ff1_w.weight.zero_(); ff1_w.bias.zero_()
        ff2_w.weight.zero_(); ff2_w.bias.zero_()
        ff1_w.weight[:d_ff, :d] = ff1.weight
        ff1_w.bias[:d_ff] = ff1.bias
        ff2_w.weight[:d, :d_ff] = ff2.weight
        ff2_w.bias[:d] = ff2.bias

    def forward_wide(x):
        h = emb_w(x)
        # Attention block
        h_n = aln1(h)
        Q = W_q_w(h_n).view(B, T, n_heads_w, d_head).transpose(1, 2)
        K = W_k_w(h_n).view(B, T, n_heads_w, d_head).transpose(1, 2)
        Vt = W_v_w(h_n).view(B, T, n_heads_w, d_head).transpose(1, 2)
        # d_head unchanged → scaling 1/√d_head unchanged
        attn_out = F.scaled_dot_product_attention(Q, K, Vt, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d_prime)
        h = h + W_o_w(attn_out)
        # FF block
        h = h + ff2_w(F.gelu(ff1_w(aln2(h))))
        return F.linear(h, emb_w.weight)

    logits_wide = forward_wide(x)

    # ========================================
    # Verification
    # ========================================
    err = (logits_orig - logits_wide).abs().max().item()
    print(f"Max |logits_orig - logits_wide| = {err:.2e}")
    assert err < 1e-4, f"FAILED: err={err}"
    print("[OK] Theorem verified: M_{d'}(x) = M_d(x) to floating point precision")

    # Check new dimensions stay zero
    h_w = emb_w(x)
    h_n = aln1(h_w)
    zero_norm = h_n[..., d:].abs().max().item()
    print(f"Max |h'[d:]| after ALN = {zero_norm:.2e}")

    # Full forward hidden state check
    Q = W_q_w(h_n).view(B, T, n_heads_w, d_head).transpose(1, 2)
    K = W_k_w(h_n).view(B, T, n_heads_w, d_head).transpose(1, 2)
    Vt = W_v_w(h_n).view(B, T, n_heads_w, d_head).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(Q, K, Vt, is_causal=True)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d_prime)
    h_w = h_w + W_o_w(attn_out)
    h_w = h_w + ff2_w(F.gelu(ff1_w(aln2(h_w))))
    zero_final = h_w[..., d:].abs().max().item()
    print(f"Max |h'[d:]| after full block = {zero_final:.2e}")
    assert zero_final < 1e-5
    print("[OK] New dimensions remain zero throughout forward pass")

    # ========================================
    # Counterexample: standard LN breaks it
    # ========================================
    sln = nn.LayerNorm(d_prime)
    with torch.no_grad():
        sln.weight[:d] = ln1.weight; sln.weight[d:] = 1.0
        sln.bias[:d] = ln1.bias; sln.bias[d:] = 0.0
    h_std = emb_w(x)
    err_ln = (sln(h_std)[..., :d] - ln1(emb(x))).abs().max().item()
    print(f"\n[FAIL] Standard LayerNorm error = {err_ln:.2e} (breaks preservation)")

    # ========================================
    # Quantify: standard LN error grows with Δ/d
    # ========================================
    print("\nStandard LN error vs width expansion ratio Δ/d:")
    for delta in [16, 32, 64, 128, 256]:
        dp = d + delta
        e_w = nn.Embedding(V, dp)
        with torch.no_grad():
            e_w.weight.zero_()
            e_w.weight[:, :d] = emb.weight
        s = nn.LayerNorm(dp)
        with torch.no_grad():
            s.weight[:d] = ln1.weight; s.weight[d:] = 1.0
            s.bias[:d] = ln1.bias; s.bias[d:] = 0.0
        err = (s(e_w(x))[..., :d] - ln1(emb(x))).abs().max().item()
        print(f"  Δ/d = {delta/d:.2f}: err = {err:.4f}")


if __name__ == '__main__':
    verify()
