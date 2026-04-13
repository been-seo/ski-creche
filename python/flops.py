def flop_single(K: int, P_emb: int, B_block: int, P_head: int) -> int:
    """FLOP per token for single-depth Snowball at depth K.

    - 2*P_emb:        embed forward (no backward — detached)
    - (2K+2)*B_block:  K-1 block forwards (no grad) + 1 forward+backward
    - 4*P_head:        one readout forward+backward
    """
    return 2 * P_emb + (2 * K + 2) * B_block + 4 * P_head


def flop_e2e(L: int, P_emb: int, B_block: int, P_head: int) -> int:
    return 6 * (P_emb + L * B_block + P_head)


def flop_ratio(K_max: int, P_emb: int, B_block: int, P_head: int) -> float:
    avg = sum(flop_single(k, P_emb, B_block, P_head) for k in range(1, K_max + 1)) / K_max
    return avg / flop_e2e(K_max, P_emb, B_block, P_head)
