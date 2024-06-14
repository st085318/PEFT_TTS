import sys
import os

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.core import adapter_mixins
from omegaconf import DictConfig, OmegaConf, open_dict

import torch


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


if __name__ == "__main__":

    pretrained_fastpitch_checkpoint = sys.argv[1]
    save_checkpoint_to = sys.argv[2]

    print(f'{pretrained_fastpitch_checkpoint} -> {save_checkpoint_to}')

    train_manifest = train_manifest = os.path.abspath(os.path.join('/workspace/baseline/NeMoTTS_dataset/LibriTTS', 'train-adapter-1.json'))
    sample_rate = 22050


    train_data = read_manifest(train_manifest)
    for m in train_data: m['audio_filepath'] = os.path.abspath(m['audio_filepath'])
    write_manifest(train_manifest, train_data)
    wave_model = WaveformFeaturizer(sample_rate=sample_rate)
    train_data = read_manifest(train_manifest)

    spec_model = FastPitchModel.restore_from(pretrained_fastpitch_checkpoint).eval().cuda()
    spec_model.cfg = update_model_config_to_support_adapter(spec_model.cfg)

    spk_embs = []
    for data in train_data:
        with torch.no_grad():
            audio = wave_model.process(data['audio_filepath'])
            audio_length = torch.tensor(audio.shape[0]).long()
            audio = audio.unsqueeze(0).to(device=spec_model.device)
            audio_length = audio_length.unsqueeze(0).to(device=spec_model.device)
            spec_ref, spec_ref_lens = spec_model.preprocessor(input_signal=audio, length=audio_length)
            spk_emb = spec_model.fastpitch.get_speaker_embedding(batch_size=spec_ref.shape[0],
                                                                speaker=None,
                                                                reference_spec=spec_ref,
                                                                reference_spec_lens=spec_ref_lens)

        spk_embs.append(spk_emb.squeeze().cpu())

    spk_embs = torch.stack(spk_embs, dim=0)
    spk_emb  = torch.mean(spk_embs, dim=0)
    spk_emb_dim = spk_emb.shape[0]

    with open_dict(spec_model.cfg):
        spec_model.cfg.speaker_encoder.precomputed_embedding_dim = spec_model.cfg.symbols_embedding_dim

    spec_model.fastpitch.speaker_encoder.overwrite_precomputed_emb(spk_emb)
    spec_model.save_to(save_checkpoint_to)