
from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import numpy as np

def cos_sim(embed_a, embed_b):
    return (embed_a @ embed_b) / (np.linalg.norm(embed_a) * np.linalg.norm(embed_b)) 

# write correct path
data_dir = Path("path to synthese audios", "dir")
wav_fpaths = list(data_dir.glob("**/*.wav"))
wavs = [preprocess_wav(wav_fpath) for wav_fpath in \
        tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit=" utterances")]


## Compute the embeddings
encoder = VoiceEncoder()
embeds = np.array([encoder.embed_utterance(wav) for wav in wavs])
speakers = np.array([fpath.parent.name for fpath in wav_fpaths])
names = np.array([fpath.stem for fpath in wav_fpaths])

speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"), 
                        lambda wav_fpath: wav_fpath.parent.stem)}

speakers_embeds = {
    speaker: encoder.embed_speaker(es) for speaker, es in speaker_wavs.items()     
} 

#print(speakers_embeds.keys())
print(f"Finetuning: {cos_sim(speakers_embeds['orig'], speakers_embeds['finetune'])}")
print(f"Adapter: {cos_sim(speakers_embeds['orig'], speakers_embeds['adapter'])}")
print(f"LoRa: {cos_sim(speakers_embeds['orig'], speakers_embeds['big_lora'])}")
print(f"ia3: {cos_sim(speakers_embeds['orig'], speakers_embeds['ia3'])}")