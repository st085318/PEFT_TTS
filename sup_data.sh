#!/usr/bin/env bash

code_dir=/workspace/
train_manifest=/workspace/NeMoTTS_dataset/LibriTTS/train-adapter-1.json
valid_manifest=/workspace/NeMoTTS_dataset/LibriTTS/val-adapter-1.json
supp_dir=/workspace/NeMoTTS_sup_data
sample_rate=22050

(HYDRA_FULL_ERROR=1 python scripts/dataset_processing/tts/extract_sup_data.py \
    manifest_filepath=$valid_manifest \
    sup_data_path=$supp_dir \
    dataset.sample_rate=$sample_rate \
    dataset.n_fft=1024 \
    dataset.win_length=1024 \
    dataset.hop_length=256 \
    +dataloader_params.num_workers=4
)

#13301 EOF