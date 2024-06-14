import os
import shutil
from pathlib import Path
import soundfile as sf
import json
import torch
from collections import defaultdict
import shutil

import pytorch_lightning as pl
import numpy as np
import tqdm

from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer


def gt_spectrogram(audio_path, wave_model, spec_gen_model):
    features = wave_model.process(audio_path, trim=False)
    audio, audio_length = features, torch.tensor(features.shape[0]).long()
    audio = audio.unsqueeze(0).to(device=spec_gen_model.device)
    audio_length = audio_length.unsqueeze(0).to(device=spec_gen_model.device)
    with torch.no_grad():
        spectrogram, spec_len = spec_gen_model.preprocessor(input_signal=audio, length=audio_length)
    return spectrogram, spec_len

def gen_spectrogram(text, spec_gen_model, speaker, reference_spec, reference_spec_lens):
    parsed = spec_gen_model.parse(text)
    speaker = torch.tensor([speaker]).long().to(device=spec_gen_model.device)
    with torch.no_grad():
        spectrogram = spec_gen_model.generate_spectrogram(tokens=parsed,
                                                          speaker=speaker,
                                                          reference_spec=reference_spec,
                                                          reference_spec_lens=reference_spec_lens)

    return spectrogram

def synth_audio(vocoder_model, spectrogram):
    with torch.no_grad():
        audio = vocoder_model.convert_spectrogram_to_audio(spec=spectrogram)
    if isinstance(audio, torch.Tensor):
        audio = audio.to('cpu').numpy()
    return audio

def main():
    sample_rate = 22050
    wave_model = WaveformFeaturizer(sample_rate=sample_rate)
    # write correct path
    fastpitch_ckpt = 'path_to_fastpitch checkpoint'
    finetuned_hifigan_on_adaptation_checkpoint = 'path to hifigan checkpoint'
    spec_model = FastPitchModel.restore_from(fastpitch_ckpt)
    spec_model = spec_model.eval().cuda()
    vocoder_model = HifiGanModel.restore_from(finetuned_hifigan_on_adaptation_checkpoint).eval().cuda()

    # write correct path
    valid_manifest = 'path to valid dataset'
    train_manifest = 'path to train dataset'
    reference_records = []
    with open(train_manifest, "r") as f:
        for i, line in enumerate(f):
            reference_records.append(json.loads(line))

    speaker_to_index = defaultdict(list)
    for i, d in enumerate(reference_records): speaker_to_index[d.get('speaker', None)].append(i)

    # Validatation Audio
    num_val = 200
    val_records = []
    with open(valid_manifest, "r") as f:
        for i, line in enumerate(f):
            val_records.append(json.loads(line))
            if len(val_records) >= num_val:
                break

    for val_record in tqdm.tqdm(val_records):
        reference_record = reference_records[speaker_to_index[val_record['speaker']][0]]
        reference_spec, reference_spec_lens = gt_spectrogram(reference_record['audio_filepath'], wave_model, spec_model)
        reference_spec = reference_spec.to(spec_model.device)
        spec_pred = gen_spectrogram(val_record['text'],
                                    spec_model,
                                    speaker=val_record['speaker'],
                                    reference_spec=reference_spec,
                                    reference_spec_lens=reference_spec_lens)

        audio_gen = synth_audio(vocoder_model, spec_pred)
        # write correct path
        sf.write(f"path/{val_record['audio_filepath'].split('/')[-1]}", np.ravel(audio_gen), sample_rate)

if __name__ == '__main__':
    main()
  