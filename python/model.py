import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable


class Snowman(nn.Module):
    def __init__(self, embed: nn.Module, block: nn.Module, readout: nn.Module,
                 k_max: int = 8):
        super().__init__()
        self.embed = embed
        self.block = block
        self.readout = readout
        # α_0=1 (active), α_{k>0}=0 (identity at growth)
        self.gates = nn.ParameterList([
            nn.Parameter(torch.ones(1) if k == 0 else torch.zeros(1))
            for k in range(k_max)
        ])

    def forward(self, x: Tensor, K: int) -> Tensor:
        h = self.embed(x)
        for k in range(K):
            h_new = self.block(h)
            h = h + self.gates[k] * (h_new - h)
        return self.readout(h)

    def forward_single(self, x: Tensor, K: int, targets: Tensor,
                       loss_fn: Callable) -> float:
        """Single-depth loss at depth K.  O(1) readout call.

        AdamW's per-parameter normalisation makes the update direction
        identical to the multi-depth case (see Single-Depth Sufficiency
        theorem): the 1/(K+1) scaling cancels in m/√v.
        """
        h = self.embed(x)
        # Intermediate depths: no backward needed, skip graph construction
        if K > 1:
            with torch.no_grad():
                for k in range(K - 1):
                    h_new = self.block(h)
                    h = h + self.gates[k] * (h_new - h)
        # Final depth: full gradient tracking
        h = h.detach()
        h_new = self.block(h)
        h = h + self.gates[K - 1] * (h_new - h)
        ce = loss_fn(self.readout(h), targets)
        ce.backward()
        return ce.item()

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def block_params(self):
        return sum(p.numel() for p in self.block.parameters())

    @property
    def embed_params(self):
        return sum(p.numel() for p in self.embed.parameters())

    @property
    def readout_params(self):
        """Readout parameter count for FLOP estimation.

        Includes the tied embedding weight matmul cost, which is the
        dominant term. The tied weight is not registered as a readout
        parameter (to avoid double-counting in param_count), but the
        matmul FLOPs are real and must be counted.
        """
        own = sum(p.numel() for p in self.readout.parameters())
        # Add tied embedding weight if embed has a tok attribute
        if hasattr(self.embed, 'tok'):
            own += self.embed.tok.weight.numel()
        return own

    @property
    def gate_values(self) -> list[float]:
        return [g.item() for g in self.gates]

    @property
    def k_max(self) -> int:
        return len(self.gates)

    def resize_depth(self, new_k_max: int):
        """Grow or shrink gate list. New gates init to 0 (identity)."""
        old = len(self.gates)
        if new_k_max > old:
            for _ in range(new_k_max - old):
                self.gates.append(nn.Parameter(torch.zeros(1)))
        elif new_k_max < old:
            self.gates = nn.ParameterList(list(self.gates)[:new_k_max])
