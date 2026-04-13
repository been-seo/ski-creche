from dataclasses import dataclass, field
from typing import Callable, Optional
import torch
import torch.nn.functional as F


def _default_loss(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


@dataclass
class SnowballConfig:
    k_max: int = None
    lr: float = None
    weight_decay: float = None
    grad_clip: float = None
    # per phase when flop_budget is None
    total_steps: int = None
    eval_interval: int = None
    log_interval: int = None
    eval_batches: int = None
    device: str = None                              # None = auto-detect
    seed: int = 42
    loss_fn: Callable = field(default_factory=lambda: _default_loss)
    optimizer_factory: Optional[Callable] = None     # None = AdamW

    # resume from deeper depth
    start_k: int = 1                                   # skip phases below this

    # checkpoint
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: Optional[int] = None        # every N steps
    checkpoint_on_phase: bool = True
    checkpoint_best: bool = True

    # db logging
    db_path: Optional[str] = None
    run_name: str = 'snowball'

    def __post_init__(self):
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def validate(self):
        required = ['k_max', 'lr', 'weight_decay', 'grad_clip',
                    'eval_interval', 'log_interval', 'eval_batches']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            raise ValueError(f"required: {', '.join(missing)}")

    def make_optimizer(self, params):
        if self.optimizer_factory is not None:
            return self.optimizer_factory(params)
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
