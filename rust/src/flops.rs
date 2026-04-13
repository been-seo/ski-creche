/// FLOP per token for single-depth Snowball at depth K.
///
/// - 2*p_emb:          embed forward (no backward — detached)
/// - (2K+2)*b_block:   K-1 block forwards (no grad) + 1 forward+backward
/// - 4*p_head:         one readout forward+backward
pub fn flop_single(k: usize, p_emb: usize, b_block: usize, p_head: usize) -> usize {
    2 * p_emb + (2 * k + 2) * b_block + 4 * p_head
}

pub fn flop_e2e(l: usize, p_emb: usize, b_block: usize, p_head: usize) -> usize {
    6 * (p_emb + l * b_block + p_head)
}

pub fn flop_ratio(k_max: usize, p_emb: usize, b_block: usize, p_head: usize) -> f64 {
    let avg: f64 = (1..=k_max)
        .map(|k| flop_single(k, p_emb, b_block, p_head) as f64)
        .sum::<f64>()
        / k_max as f64;
    avg / flop_e2e(k_max, p_emb, b_block, p_head) as f64
}
