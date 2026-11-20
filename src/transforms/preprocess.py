import torch
import torchaudio

# Preprocessing Audio

BASE_SR = 16000

def preprocess(waveform, sample_rate):
    # Make audio mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Equalize sample rates
    if sample_rate != BASE_SR:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=BASE_SR)

    # Normalize audio

    return waveform, BASE_SR