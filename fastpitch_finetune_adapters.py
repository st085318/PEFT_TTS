import os
from dataclasses import is_dataclass
import shutil
from pathlib import Path
import time

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.core import adapter_mixins
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def update_model_config_to_support_adapter(config) -> DictConfig:
    with open_dict(config):
        enc_adapter_metadata = adapter_mixins.get_registered_adapter(config.input_fft._target_)
        if enc_adapter_metadata is not None:
            config.input_fft._target_ = enc_adapter_metadata.adapter_class_path

        dec_adapter_metadata = adapter_mixins.get_registered_adapter(config.output_fft._target_)
        if dec_adapter_metadata is not None:
            config.output_fft._target_ = dec_adapter_metadata.adapter_class_path

        pitch_predictor_adapter_metadata = adapter_mixins.get_registered_adapter(config.pitch_predictor._target_)
        if pitch_predictor_adapter_metadata is not None:
            config.pitch_predictor._target_ = pitch_predictor_adapter_metadata.adapter_class_path

        duration_predictor_adapter_metadata = adapter_mixins.get_registered_adapter(config.duration_predictor._target_)
        if duration_predictor_adapter_metadata is not None:
            config.duration_predictor._target_ = duration_predictor_adapter_metadata.adapter_class_path

        aligner_adapter_metadata = adapter_mixins.get_registered_adapter(config.alignment_module._target_)
        if aligner_adapter_metadata is not None:
            config.alignment_module._target_ = aligner_adapter_metadata.adapter_class_path

    return config


def add_global_adapter_cfg(model, global_adapter_cfg):
    # Convert to DictConfig from dict or Dataclass
    if is_dataclass(global_adapter_cfg):
        global_adapter_cfg = OmegaConf.structured(global_adapter_cfg)

    if not isinstance(global_adapter_cfg, DictConfig):
        global_adapter_cfg = DictConfig(global_adapter_cfg)

    # Update the model.cfg with information about the new adapter global cfg
    with open_dict(global_adapter_cfg), open_dict(model.cfg):
        if 'adapters' not in model.cfg:
            model.cfg.adapters = OmegaConf.create({})

        # Add the global config for adapters to the model's internal config
        model.cfg.adapters[model.adapter_global_cfg_key] = global_adapter_cfg

        # Update all adapter modules (that already exist) with this global adapter config
        model.update_adapter_cfg(model.cfg.adapters)


@hydra_runner(config_path="conf", config_name="fastpitch_align_adapter")
def main(cfg):
    if hasattr(cfg.model.optim, 'sched'):
        logging.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")

    trainer = pl.Trainer(**cfg.trainer)
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    # Initialize FastPitchModel
    model = FastPitchModel(cfg=update_model_config_to_support_adapter(cfg.model), trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)

    # Extract adapter parameters
    with open_dict(cfg.model.adapter):
        # Extract the name of the adapter (must be given for training)
        adapter_name = cfg.model.adapter.pop("adapter_name", "adapter")
        # Extract the name of the modules where adapters need to be added (must be given for training)
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)

        adapter_submodule_name = cfg.model.adapter.pop("adapter_submodule_name", None)
        # Name of the adapter checkpoint which will be saved after training
        adapter_state_dict_name = cfg.model.adapter.pop("adapter_state_dict_name", None)

        # augment adapter name with module name, if not provided by user
        if adapter_submodule_name is not None and ':' not in adapter_name:
            adapter_name = f'{adapter_submodule_name}_{adapter_name}'
            if adapter_module_name is not None:
                adapter_name = f'{adapter_module_name}:{adapter_name}' 
        elif adapter_module_name is not None and ':' not in adapter_name:
            adapter_name = f'{adapter_module_name}:{adapter_name}'

        # Extract the global adapter config, if provided
        adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)

    # Freeze model
    if cfg.get('model_freeze', False):
        model.freeze()
    if cfg.get('embeddding_unfreeze', False):
        for name, param in model.fastpitch.speaker_encoder.named_parameters():
            param.requires_grad = True

    # Setup adapters
    if adapter_global_cfg is not None:
        add_global_adapter_cfg(model, adapter_global_cfg)

    if cfg.model.get("unfreeze_aligner", False):
        for name, param in model.fastpitch.aligner.named_parameters():
            param.requires_grad = True

    if cfg.model.get("unfreeze_duration_predictor", False):
        for name, param in model.fastpitch.duration_predictor.named_parameters():
            param.requires_grad = True

    if cfg.model.get("unfreeze_pitch_predictor", False):
        for name, param in model.fastpitch.pitch_predictor.named_parameters():
            param.requires_grad = True

    if cfg.model.adapter.get('speaker', -1) != -1:
        adapter_name = f'{adapter_name}_{cfg.model.adapter.speaker}'
        multispeaker_adapter_names = [adapter_name]
    else:
        multispeaker_adapter_names = [adapter_name + f'_{i}' for i in range(cfg.model.speaker_encoder.lookup_module.n_speakers)]

    # Add adapters
    if cfg.get('adapter_load', False):
        model.load_adapters(f'{cfg.exp_manager.exp_dir}/FastPitch/adapters.pt')
    elif cfg.model.adapter.get('speaker', -1) != -1:
        adapter_config = cfg.model.adapter
        model.add_adapter(name=adapter_name, cfg=adapter_config)
    else:
        for i in range(len(multispeaker_adapter_names)):
            adapter_config = cfg.model.adapter
            adapter_config.speaker = i
            model.add_adapter(name=multispeaker_adapter_names[i], cfg=adapter_config)
    assert model.is_adapter_available()
    # enable adapters
    model.set_inner_adapters(model.get_enabled_adapters())
    model.set_enabled_adapters(enabled=False)
    #print(f'Enabled adapters {str(model.get_enabled_adapters())}')
    for name in multispeaker_adapter_names:
        model.set_enabled_adapters(name, enabled=True)
    model.unfreeze_enabled_adapters()
    if cfg.model.adapter.get('speaker', -1) == -1:
        model.freeze_enabled_adapters()

    # Set model to training mode.
    model = model.train()
    # Then, Unfreeze just the adapter weights that were enabled above (no part of model)
    model.unfreeze_enabled_adapters()
    # summarize the model
    model.summarize()

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)

    model.set_enabled_adapters(enabled=True)

    # Save the adapter state dict after training has completed
    if adapter_state_dict_name is not None:
        state_path = exp_log_dir if exp_log_dir is not None else os.getcwd()
        ckpt_path = os.path.join(state_path, "checkpoints")
        if os.path.exists(ckpt_path):
            state_path = ckpt_path

        # Save the adapter modules in a seperate file
        model.save_adapters(os.path.join(state_path, adapter_state_dict_name))

    logs_dir = cfg.exp_manager.exp_dir
    last_checkpoint_dir = sorted(list([i for i in (Path(logs_dir) / "FastPitch").iterdir() if i.is_dir()]))[-1] / "checkpoints"
    pretrained_fastpitch_checkpoint = os.path.abspath(list(last_checkpoint_dir.glob('*.nemo'))[0])
    

    dst = '/'.join(pretrained_fastpitch_checkpoint.split('/')[:-3] +
        pretrained_fastpitch_checkpoint.split('/')[-1:])
    shutil.copyfile(pretrained_fastpitch_checkpoint, dst)
    dst = '/'.join(pretrained_fastpitch_checkpoint.split('/')[:-3] +
        ['adapters.pt'])
    src = pretrained_fastpitch_checkpoint.split('/')
    src[-1] = 'adapters.pt'
    src = '/'.join(src)
    shutil.copyfile(src, dst)



if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
