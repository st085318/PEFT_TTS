#!/usr/bin/env bash

#change paths and pitch data
code_dir=/workspace
train_manifest=/workspace/NeMoTTS_dataset/LibriTTS/train-adapter-1.json
valid_manifest=/workspace/NeMoTTS_dataset/LibriTTS/val-adapter-1.json
supp_dir=/workspace/NeMoTTS_sup_data
PITCH_MEAN=178.7997589111328
PITCH_STD=18.972787857055664
phoneme_dict_path=/workspace/scripts/tts_dataset_files/cmudict-0.7b_nv22.10
heteronyms_path=/workspace/scripts/tts_dataset_files/heteronyms-052722
logs_dir=/workspace/NeMoTTS_logs_model/add_speaker/adapter
pretrain_ckpt=/workspace/NeMoTTS_logs_model/FastPitch/FastPitch_add_speaker_emb1.nemo
finetune_checkpoint=/workspace/NeMoTTS_logs_model/add_speaker/adapter/FastPitch/FastPitch.nemo

(HYDRA_FULL_ERROR=1 find_unused_parameters=True python fastpitch_finetune.py \
--config-name=fastpitch_align_22050_adapter.yaml \
+init_from_nemo_model=$pretrain_ckpt \
train_dataset=$train_manifest \
validation_datasets=$valid_manifest \
sup_data_types="['align_prior_matrix', 'pitch', 'energy']" \
sup_data_path=$supp_dir \
pitch_mean=$PITCH_MEAN \
pitch_std=$PITCH_STD \
model.speaker_encoder.precomputed_embedding_dim=384 \
~model.speaker_encoder.lookup_module \
~model.speaker_encoder.gst_module \
model.train_ds.dataloader_params.batch_size=16 \
model.validation_ds.dataloader_params.batch_size=16 \
model.optim.name=adam \
model.optim.lr=1e-3 \
model.optim.sched.warmup_steps=100 \
exp_manager.exp_dir=$logs_dir \
+exp_manager.create_wandb_logger=True \
+exp_manager.wandb_logger_kwargs.name="finetune" \
+exp_manager.wandb_logger_kwargs.project="voice_cloning" \
+exp_manager.checkpoint_callback_params.save_top_k=-1 \
trainer.max_epochs=1500 \
trainer.check_val_every_n_epoch=50 \
trainer.log_every_n_steps=50 \
trainer.devices=-1 \
trainer.strategy=ddp \
trainer.precision=16 \
)