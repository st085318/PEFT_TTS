# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import dataclasses
import inspect
from bisect import bisect
import math
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass

import hydra
import torch.optim as optim
import torch.optim.lr_scheduler as pt_scheduler
import torch.utils.data.dataloader as dataloader
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler

from nemo.core.config import SchedulerParams, get_scheduler_config, register_scheduler_params
from nemo.utils import logging
from nemo.utils.model_utils import maybe_update_config_version


class MultiStepLR(_LRScheduler):
    def __init__(
        self, optimizer, *, milestones = [], gamma=0.1, warmup_steps=0, min_lr=0.0, last_epoch=-1
    ):
        self.gamma = gamma
        self.milestones = milestones
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the minimum learning rate."
                )

        new_lrs = [self.lr(initial_lr=initial_lr, step=step) for initial_lr in self.base_lrs]
        return new_lrs

    def lr(self, initial_lr, step):
        if self.warmup_steps > step:
            mult = step / self.warmup_steps ** 1.5
        else:
            mult = self.gamma ** bisect(self.milestones, step)

        out_lr = initial_lr * mult
        if step > self.warmup_steps:
            out_lr = max(out_lr, self.min_lr)
        return out_lr

@dataclass
class MultiStepLRParams(SchedulerParams):
    #optimizer: Any
    milestones: Tuple = ()
    gamma: float = 0.5
    warmup_steps: int = 0
    min_lr: float = 0.0