import torch
import torchaudio

# Applying augmentation

snr = torch.tensor([30])

def augment(sample, sample_rate):
    wf_mix = sample.get("mix", None)
    wf_s1 = sample.get("s1", None)
    wf_s2 = sample.get("s2", None)

    

    return sample