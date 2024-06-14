#!/usr/bin/env bash

code_dir=/workspace/baseline/NeMoTTS
train_manifest=/workspace/baseline/NeMoTTS_dataset/LibriTTS/train_360.json
valid_manifest=/workspace/baseline/NeMoTTS_dataset/LibriTTS/val_360.json
#train_manifest=/workspace/baseline/NeMoTTS_dataset/LibriTTS/train-adapter-1.json
#valid_manifest=/workspace/baseline/NeMoTTS_dataset/LibriTTS/val-adapter-1.json
supp_dir=/workspace/baseline/NeMoTTS_sup_data
#PITCH_MEAN=164.57972717285156
#PITCH_STD=65.4063949584961
#PITCH_MEAN=166.03884887695312
#PITCH_STD=70.57227325439453
PITCH_MEAN=171.05506896972656 # train ds
PITCH_STD=76.94561004638672
#PITCH_MEAN=178.7997589111328
#PITCH_STD=18.972787857055664
phoneme_dict_path=/workspace/baseline/NeMoTTS/scripts/tts_dataset_files/cmudict-0.7b_nv22.10
heteronyms_path=/workspace/baseline/NeMoTTS/scripts/tts_dataset_files/heteronyms-052722
logs_dir=/workspace/baseline/NeMoTTS_logs_model
pretrain_ckpt=/workspace/baseline/NeMoTTS_logs_model/FastPitch/FastPitch.nemo
#pretrain_ckpt=/workspace/baseline/NeMoTTS_logs_model/FastPitch/FastPitch_add_speaker_emb.nemo
max_epochs=1000
batch_size=25
precision=16
# examples/tts/

# +model.optim.sched.min_lr=1e-3 \
(HYDRA_FULL_ERROR=1 python fastpitch.py \
--config-name=fastpitch_align_22050_adapter.yaml \
train_dataset=$train_manifest \
validation_datasets=$valid_manifest \
sup_data_types="['align_prior_matrix', 'pitch', 'speaker_id', 'reference_audio']" \
sup_data_path=$supp_dir \
pitch_mean=$PITCH_MEAN \
pitch_std=$PITCH_STD \
phoneme_dict_path=$phoneme_dict_path \
heteronyms_path=$heteronyms_path \
model.speaker_encoder.lookup_module.n_speakers=100 \
model.input_fft.condition_types="['add', 'layernorm']" \
model.output_fft.condition_types="['add', 'layernorm']" \
model.duration_predictor.condition_types="['add', 'layernorm']" \
model.pitch_predictor.condition_types="['add', 'layernorm']" \
model.alignment_module.condition_types="['add']" \
model.train_ds.dataloader_params.batch_size=$batch_size \
model.train_ds.dataloader_params.num_workers=1 \
model.validation_ds.dataloader_params.batch_size=$batch_size \
model.validation_ds.dataloader_params.num_workers=1 \
model.train_ds.dataset.max_duration=20 \
model.validation_ds.dataset.max_duration=20 \
model.optim.lr=2e-4 \
exp_manager.exp_dir=$logs_dir \
+exp_manager.create_wandb_logger=True \
+exp_manager.wandb_logger_kwargs.name="Fastpitch_TemporalTransformer" \
+exp_manager.wandb_logger_kwargs.project="voice_cloning_pretrain" \
trainer.max_epochs=$max_epochs \
trainer.check_val_every_n_epoch=20 \
trainer.log_every_n_steps=20 \
trainer.devices=-1 \
trainer.strategy=ddp \
trainer.precision=$precision
)

#+exp_manager.wandb_logger_kwargs.project="voice_cloning_pretrain" \
# model.speaker_encoder.lookup_module.n_speakers=100 \
# (HYDRA_FULL_ERROR=1 python fastpitch.py \
# --config-name=fastpitch_align_22050_adapter.yaml \
# +init_from_nemo_model=$pretrain_ckpt \
# train_dataset=$train_manifest \
# validation_datasets=$valid_manifest \
# sup_data_types="['align_prior_matrix', 'pitch', 'speaker_id', 'reference_audio']" \
# sup_data_path=$supp_dir \
# pitch_mean=$PITCH_MEAN \
# pitch_std=$PITCH_STD \
# phoneme_dict_path=$phoneme_dict_path \
# heteronyms_path=$heteronyms_path \
# ~model.speaker_encoder.lookup_module \
# ~model.speaker_encoder.gst_module \
# model.input_fft.condition_types="['add', 'layernorm']" \
# model.output_fft.condition_types="['add', 'layernorm']" \
# model.duration_predictor.condition_types="['add', 'layernorm']" \
# model.pitch_predictor.condition_types="['add', 'layernorm']" \
# model.alignment_module.condition_types="['add']" \
# model.train_ds.dataloader_params.batch_size=$batch_size \
# model.train_ds.dataloader_params.num_workers=10 \
# model.validation_ds.dataloader_params.batch_size=$batch_size \
# model.validation_ds.dataloader_params.num_workers=10 \
# model.train_ds.dataset.max_duration=20 \
# model.validation_ds.dataset.max_duration=20 \
# model.optim.name=adam \
# model.optim.lr=1e-3 \
# exp_manager.exp_dir=$logs_dir \
# +exp_manager.create_wandb_logger=True \
# +exp_manager.wandb_logger_kwargs.name="Fastpitch_finetune_sp_1" \
# +exp_manager.wandb_logger_kwargs.project="voice_cloning" \
# trainer.max_epochs=$max_epochs \
# trainer.check_val_every_n_epoch=20 \
# trainer.log_every_n_steps=20 \
# trainer.devices=-1 \
# trainer.strategy=ddp \
# trainer.precision=$precision
# )

# (HYDRA_FULL_ERROR=1 python fastpitch.py \
# --config-name=fastpitch_align_22050_adapter.yaml \
# +init_from_nemo_model=$pretrain_ckpt \
# train_dataset=$train_manifest \
# validation_datasets=$valid_manifest \
# sup_data_types="['align_prior_matrix', 'pitch', 'speaker_id', 'reference_audio']" \
# sup_data_path=$supp_dir \
# pitch_mean=$PITCH_MEAN \
# pitch_std=$PITCH_STD \
# phoneme_dict_path=$phoneme_dict_path \
# heteronyms_path=$heteronyms_path \
# model.speaker_encoder.lookup_module.n_speakers=100 \
# model.input_fft.condition_types="['add', 'layernorm']" \
# model.output_fft.condition_types="['add', 'layernorm']" \
# model.duration_predictor.condition_types="['add', 'layernorm']" \
# model.pitch_predictor.condition_types="['add', 'layernorm']" \
# model.alignment_module.condition_types="['add']" \
# model.train_ds.dataloader_params.batch_size=$batch_size \
# model.train_ds.dataloader_params.num_workers=10 \
# model.validation_ds.dataloader_params.batch_size=$batch_size \
# model.validation_ds.dataloader_params.num_workers=10 \
# model.train_ds.dataset.max_duration=20 \
# model.validation_ds.dataset.max_duration=20 \
# model.optim.lr=2e-4 \
# +model.optim.sched.min_lr=2e-4 \
# exp_manager.exp_dir=$logs_dir \
# +exp_manager.create_wandb_logger=True \
# +exp_manager.wandb_logger_kwargs.name="Fastpitch_3" \
# +exp_manager.wandb_logger_kwargs.project="voice_cloning_pretrain" \
# trainer.max_epochs=$max_epochs \
# trainer.check_val_every_n_epoch=20 \
# trainer.log_every_n_steps=20 \
# trainer.devices=-1 \
# trainer.strategy=ddp \
# trainer.precision=$precision
# )

# (HYDRA_FULL_ERROR=1 python fastpitch.py \
# --config-name=fastpitch_align_22050_adapter.yaml \
# +init_from_nemo_model=$pretrain_ckpt \
# train_dataset=$train_manifest \
# validation_datasets=$valid_manifest \
# sup_data_types="['align_prior_matrix', 'pitch', 'speaker_id', 'reference_audio']" \
# sup_data_path=$supp_dir \
# pitch_mean=$PITCH_MEAN \
# pitch_std=$PITCH_STD \
# phoneme_dict_path=$phoneme_dict_path \
# heteronyms_path=$heteronyms_path \
# model.speaker_encoder.lookup_module.n_speakers=61 \
# model.input_fft.condition_types="['add', 'layernorm']" \
# model.output_fft.condition_types="['add', 'layernorm']" \
# model.duration_predictor.condition_types="['add', 'layernorm']" \
# model.pitch_predictor.condition_types="['add', 'layernorm']" \
# model.alignment_module.condition_types="['add']" \
# model.train_ds.dataloader_params.batch_size=$batch_size \
# model.train_ds.dataloader_params.num_workers=10 \
# model.validation_ds.dataloader_params.batch_size=$batch_size \
# model.validation_ds.dataloader_params.num_workers=10 \
# model.train_ds.dataset.max_duration=20 \
# model.validation_ds.dataset.max_duration=20 \
# model.optim.lr=2e-6 \
# +model.optim.sched.min_lr=2e-6 \
# exp_manager.exp_dir=$logs_dir \
# +exp_manager.create_wandb_logger=True \
# +exp_manager.wandb_logger_kwargs.name="Fastpitch_4" \
# +exp_manager.wandb_logger_kwargs.project="voice_cloning_pretrain" \
# trainer.max_epochs=$max_epochs \
# trainer.check_val_every_n_epoch=20 \
# trainer.log_every_n_steps=20 \
# trainer.devices=-1 \
# trainer.strategy=ddp \
# trainer.precision=$precision
# )