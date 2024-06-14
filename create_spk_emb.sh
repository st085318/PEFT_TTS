#!/usr/bin/env bash

pretrain_ckpt=/workspace/baseline/NeMoTTS_logs_model/FastPitch/FastPitch.nemo
save_to=/workspace/baseline/NeMoTTS_logs_model/FastPitch/FastPitch_add_speaker_emb.nemo

python create_spk_emb.py $pretrain_ckpt $save_to
