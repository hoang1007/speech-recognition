import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        feature_size: int,
        factor: float = 1.0,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.feature_size = feature_size
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self._compute_lr()
        return [lr] * len(self.base_lrs)

    def _compute_lr(self):
        if self.last_epoch == 0:
            return 0.0

        lr = (self.feature_size ** (-0.5)) * min(
            self.last_epoch ** (-0.5), self.last_epoch * self.warmup_steps ** (-1.5)
        )

        return lr * self.factor


class TriStateScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        constant_steps: int,
        factor: float = 0.3,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.total_steps = total_steps
        self.factor = factor

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not hasattr(self, "eta_min"):
            self.eta_max = self.base_lrs.copy()
            self.eta_min = [eta_max * self.factor for eta_max in self.eta_max]

        return [
            self._compute_lr(group["lr"], eta_min, eta_max)
            for group, eta_min, eta_max in zip(
                self.optimizer.param_groups, self.eta_min, self.eta_max
            )
        ]

    def _compute_lr(self, prev_lr: float, eta_min: float, eta_max: float):
        # first stage
        if self.last_epoch <= self.warmup_steps:
            lr = eta_max - 0.5 * (eta_max - eta_min) * (
                1 + math.cos(math.pi * self.last_epoch / self.warmup_steps)
            )
        # second stage
        elif self.last_epoch <= self.warmup_steps + self.constant_steps:
            lr = prev_lr
        else:
            # third stage
            decay_steps = self.total_steps - self.warmup_steps - self.constant_steps
            k = self.last_epoch - self.warmup_steps - self.constant_steps
            lr = eta_min + 0.5 * (eta_max - eta_min) * (
                1 + math.cos(math.pi * k / decay_steps)
            )

        return lr

    def state_dict(self) -> dict:
        return super().state_dict()
