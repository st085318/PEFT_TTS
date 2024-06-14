# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import os
from dataclasses import is_dataclass
import shutil
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo.core.optim.lr_scheduler import register_scheduler
from lr_schedulers import MultiStepLR, MultiStepLRParams


@hydra_runner(config_path="conf", config_name="fastpitch_align_44100_adapter")
def main(cfg):
    register_scheduler('MultiStepLR', MultiStepLR, MultiStepLRParams)
    if hasattr(cfg.model.optim, 'sched'):
        logging.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")

    trainer = pl.Trainer(**cfg.trainer)
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    # Initialize FastPitchModel
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    model = model.train()
    # Then, Unfreeze just the adapter weights that were enabled above (no part of model)
    #model.unfreeze_enabled_adapters()
    # summarize the model
    model.summarize()

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])


    trainer.fit(model)

    logs_dir = cfg.exp_manager.exp_dir
    last_checkpoint_dir = sorted(list([i for i in (Path(logs_dir) / "FastPitch").iterdir() if i.is_dir()]))[-1] / "checkpoints"
    pretrained_fastpitch_checkpoint = os.path.abspath(list(last_checkpoint_dir.glob('*.nemo'))[0])
    

    dst = '/'.join(pretrained_fastpitch_checkpoint.split('/')[:-3] +
        pretrained_fastpitch_checkpoint.split('/')[-1:])
    shutil.copyfile(pretrained_fastpitch_checkpoint, dst)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
