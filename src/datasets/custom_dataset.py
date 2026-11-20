
# Implementation of Custom Dataset for Audio preprocessing

import numpy as np
from pathlib import Path
import random
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, audio_dir, preprocess=None, augmentation=None, mode="train", dynamic_mixing_prob=0.0):
        super().__init__()
        self.audio_dir = Path(audio_dir) # audio_dir is the root of split on mix, s1, s2 
        self.preprocess = preprocess
        self.augmentation = augmentation
        self.mode = mode
        self.dynamic_mixing_prob = dynamic_mixing_prob

        self.data = []

        if mode == "test":
            for file_name in sorted((self.audio_dir / "mix").iterdir()):
                self.data.append({"mix": file_name})

        if mode == "train" or mode == "validation":
            for file_name in sorted((self.audio_dir / "mix").iterdir()):
                target_file_1 = self.audio_dir / "s1" / file_name.name
                target_file_2 = self.audio_dir / "s2" / file_name.name
                self.data.append({"mix": file_name, "s1": target_file_1, "s2": target_file_2})

    # dynamic mixing
    def make_dynamic_sample(self):
        rng = np.random.default_rng()
        source_id_1, source_id_2 = rng.choice(len(self.data), size=2, replace=False)
        wf_1, sr_1 = self.data[source_id_1]
        wf_2, sr_2 = self.data[source_id_2]

        if self.preprocess is not None:
            wf_1, sr_1 = self.preprocess(wf_1, sr_1)
            wf_2, _ = self.preprocess(wf_2, sr_2)

        return {"mix": wf_1 + wf_2, "s1": wf_1, "s2": wf_2, "sample_rate": sr_1}

    def __getitem__(self, idx):
        # Augmentation with dynamic mixing
        if random.random() > self.dynamic_mixing_prob:
            return self.make_dynamic_sample()

        item = self.data[idx]

        waveform, sample_rate = torchaudio.load(item["mix"])

        if self.preprocess:
            waveform, sample_rate = self.preprocess(waveform, sample_rate)

        if self.mode == "train" or self.mode == "validation":
            target_wf_1, target_sr_1 = torchaudio.load(item["s1"])
            target_wf_2, target_sr_2 = torchaudio.load(item["s2"])

            if self.preprocess:
                target_wf_1, _ = self.preprocess(target_wf_1, target_sr_1)
                target_wf_2, _ = self.preprocess(target_wf_2, target_sr_2)

            sample = {"mix": waveform, "s1": target_wf_1, "s2": target_wf_2, "sample_rate": sample_rate}
            if self.mode == "train" and self.augmentation is not None:
                sample = self.augmentation(sample)

            return sample

        return {"mix": waveform, "sample_rate": sample_rate}

    def __len__(self):
        return len(self.data)

# Implementation of VideoDataset for video data preprocessing
class VideoDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self):
        pass
    
    def __len__(self):
        pass