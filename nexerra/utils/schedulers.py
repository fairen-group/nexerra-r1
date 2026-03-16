# ---------------------------------------
#     _   __                              
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ / 
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/  
#
# Standard schedulers for training; all standard-implementations. 
# Author: Dhruv Menon (dm958[at]cam.ac.uk)
# 
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import math
from torch.optim.lr_scheduler import _LRScheduler

# --- Scheduler(s) for KL annealing & LR ---
class LinearBetaScheduler:
    '''Linearly anneal beta from 0 to max_beta over warmup steps'''
    def __init__(self, warmup_steps: int, max_beta: float = 0.1):
        self.warmup_steps = max(1, warmup_steps)
        self.max_beta = max_beta
    def get_beta(self, step: int) -> float:
        if step <= self.warmup_steps: return self.max_beta * (float(step) / self.warmup_steps)
        else: return self.max_beta

class RampBetaScheduler:
    '''Keep beta 0 for warmup_steps (ideally 2 epochs), then ramp from 0.0 to max_beta over ramp_steps (say 5 epochs)'''
    def __init__(self, warmup_steps: int = 7000, ramp_steps: int = 500000, max_beta: float = 0.1):
        self.warmup_steps = max(1, warmup_steps)
        self.ramp_steps = max(1, ramp_steps)
        self.max_beta = max_beta
        self.total_ramp = self.warmup_steps + self.ramp_steps
    
    def get_beta(self, step: int) -> float:
        if step < self.warmup_steps: return 0.0
        elif step < self.total_ramp:
            frac = (step - self.warmup_steps) / self.ramp_steps
            return self.max_beta * frac
        else: return self.max_beta

class CyclicalBetaScheduler:
    '''The 0 -> max_beta -> 0 cycle'''
    def __init__(self, cycle_length: int, max_beta: float = 0.1):
        assert 0.0 < max_beta <= 1.0, "max_beta must be in (0.0, 1.0)"
        self.cycle_length = cycle_length
        self.max_beta = max_beta
    
    def get_beta(self, step: int) -> float:
        pos = step % self.cycle_length
        half_cycle = self.cycle_length // 2

        if pos < half_cycle: return self.max_beta * (pos / half_cycle)
        else: return self.max_beta * (1 - (pos - half_cycle) / half_cycle)

class PiecewiseLinearBetaScheduler:
    '''0 -> 0.1 for warmup_steps, 0.1 -> 1 over ramp_steps, then hold at 1 to train on full ELBO'''
    def __init__(self, warmup_steps: int = 1000, ramp_steps: int = 500000):
        self.warmup_steps = max(1, warmup_steps)
        self.ramp_steps = max(1, ramp_steps)
        self.total_ramp = self.warmup_steps + self.ramp_steps
    def get_beta(self, step: int) -> float:
        if step < self.warmup_steps: return (step / self.warmup_steps) * 0.1
        elif self.warmup_steps < step < self.total_ramp:
            frac = (step - self.warmup_steps) / self.ramp_steps
            return 0.1 + frac * 0.9
        else: return 1.0

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

class DropoutScheduler:
    '''Linearly anneal dropout from its starting value to 0.0 over specified steps'''
    def __init__(self, decay_start: int, start_value: float, end_value: float, total_steps: int):
        self.decay_start = decay_start
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_dropout(self, step: int) -> float:
        if step >= self.decay_start:
            if step < self.total_steps:
                return self.start_value + (self.end_value - self.start_value) * (step / self.total_steps)
            else: return self.end_value
        else: return self.start_value