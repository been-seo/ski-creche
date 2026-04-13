import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Optional
from .model import Snowman
from .config import SnowballConfig
from .interfaces import DataStream
from .flops import flop_single
from .db import TrainLogger


class SnowballTrainer:
    def __init__(self, model: Snowman, config: SnowballConfig,
                 train_data: DataStream, eval_data: Optional[DataStream] = None,
                 tokens_per_step: Optional[int] = None,
                 on_step: Optional[Callable] = None,
                 on_eval: Optional[Callable] = None,
                 on_phase: Optional[Callable] = None):
        config.validate()
        self.model = model.to(config.device)
        self.cfg = config
        self.train_data = train_data
        self.eval_data = eval_data
        self.tps = tokens_per_step
        self.on_step = on_step
        self.on_eval = on_eval
        self.on_phase = on_phase

        self.P_emb = model.embed_params
        self.B_block = model.block_params
        self.P_head = model.readout_params

        self.logger = None
        if config.db_path:
            self.logger = TrainLogger(config.db_path, config)

    def _steps_per_phase(self, flop_budget: float) -> int:
        # equal steps per phase: budget = S * tps * Σ flop_single(k)
        total_cost = sum(
            flop_single(k, self.P_emb, self.B_block, self.P_head)
            for k in range(1, self.cfg.k_max + 1)
        )
        return int(flop_budget / (self.tps * total_cost))

    def _cosine_lr(self, phase_step: int, phase_total: int) -> float:
        return self.cfg.lr * 0.5 * (1 + np.cos(np.pi * phase_step / phase_total))

    def _save_checkpoint(self, step, K, opt, flops_cum, tag=''):
        cfg = self.cfg
        if not cfg.checkpoint_dir:
            return None
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        name = f'step{step}_K{K}{tag}.pt'
        path = os.path.join(cfg.checkpoint_dir, name)
        torch.save({
            'step': step, 'K': K, 'flops_cum': flops_cum,
            'model': self.model.state_dict(),
            'optimizer': opt.state_dict(),
        }, path)
        if self.logger:
            self.logger.log_checkpoint(cfg.run_name, step, K, path)
            self.logger.commit()
        return path

    @torch.no_grad()
    def evaluate(self, K: int) -> float:
        self.model.eval()
        total = 0.0
        for _ in range(self.cfg.eval_batches):
            x, y = self.eval_data.get_batch()
            logits = self.model(x, K)
            total += self.cfg.loss_fn(logits, y).item()
        self.model.train()
        return total / self.cfg.eval_batches

    def train(self, flop_budget: Optional[float] = None) -> dict:
        torch.manual_seed(self.cfg.seed)
        cfg = self.cfg
        run = cfg.run_name
        use_flop = flop_budget is not None and self.tps is not None

        model = self.model
        model.train()
        opt = cfg.make_optimizer(model.parameters())

        t0 = time.time()
        flops_cum = 0.0
        best_loss = float('inf')
        step = 0
        current_K = cfg.start_k
        phase_step = 0

        if use_flop:
            phase_total = self._steps_per_phase(flop_budget)
        else:
            if cfg.total_steps is None:
                raise ValueError("need flop_budget or total_steps")
            phase_total = cfg.total_steps

        if self.logger:
            self.logger.log_phase(run, current_K, step, flops_cum)
            self.logger.commit()

        while True:
            if phase_step >= phase_total:
                if current_K >= cfg.k_max:
                    break
                old_K = current_K
                current_K += 1
                phase_step = 0
                opt = cfg.make_optimizer(model.parameters())
                if cfg.checkpoint_on_phase and cfg.checkpoint_dir:
                    self._save_checkpoint(step, current_K, opt, flops_cum, '_phase')
                if self.logger:
                    self.logger.log_phase(run, current_K, step, flops_cum)
                    self.logger.commit()
                if self.on_phase:
                    self.on_phase(old_K, current_K, step, flops_cum)

            lr = self._cosine_lr(phase_step, phase_total)
            for pg in opt.param_groups:
                pg['lr'] = lr

            x, y = self.train_data.get_batch()
            opt.zero_grad()
            # random depth sampling: k ~ U(1..current_K)
            # E[grad] ∝ multi-depth grad; AdamW m/√v normalises the scale
            k = torch.randint(1, current_K + 1, (1,)).item()
            ce = model.forward_single(x, k, y, cfg.loss_fn)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            if self.tps is not None:
                fpt = flop_single(k, self.P_emb, self.B_block, self.P_head)
                flops_cum += fpt * self.tps

            if step % cfg.log_interval == 0:
                gates = model.gate_values[:current_K]
                if self.logger:
                    self.logger.log_step(run, step, current_K, ce,
                                         lr, flops_cum, gates, [ce])
                    self.logger.commit()
                if self.on_step:
                    self.on_step(step, current_K, ce, lr, flops_cum)

            if (step + 1) % cfg.eval_interval == 0 and self.eval_data is not None:
                val_loss = self.evaluate(current_K)
                if val_loss < best_loss:
                    best_loss = val_loss
                    if cfg.checkpoint_best and cfg.checkpoint_dir:
                        self._save_checkpoint(step + 1, current_K, opt, flops_cum, '_best')
                if self.logger:
                    self.logger.log_eval(run, step + 1, current_K, val_loss, flops_cum)
                    self.logger.commit()
                if self.on_eval:
                    self.on_eval(step + 1, current_K, val_loss, flops_cum)

            if cfg.checkpoint_interval and cfg.checkpoint_dir:
                if (step + 1) % cfg.checkpoint_interval == 0:
                    self._save_checkpoint(step + 1, current_K, opt, flops_cum)

            step += 1
            phase_step += 1

        elapsed = time.time() - t0

        final_loss = None
        if self.eval_data is not None:
            final_loss = self.evaluate(cfg.k_max)
            if final_loss < best_loss:
                best_loss = final_loss

        result = {
            'best_loss': best_loss,
            'final_loss': final_loss if final_loss is not None else ce,
            'total_steps': step,
            'elapsed': elapsed,
            'total_flops': flops_cum,
            'params': model.param_count,
            'block_params': model.block_params,
        }

        if self.logger:
            for k, v in result.items():
                self.logger.log_summary(run, k, v)
            self.logger.commit()
            self.logger.close()

        return result
