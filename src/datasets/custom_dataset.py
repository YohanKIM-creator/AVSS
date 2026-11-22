
# Implementation of Custom Dataset for Audio preprocessing

import numpy as np
from pathlib import Path
import random
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, audio_dir, preprocess=None, augmentation=None, mouths_dir=None, mode="train", dynamic_mixing_prob=0.0):
        super().__init__()
        self.audio_dir = Path(audio_dir) # audio_dir is the root of split on mix, s1, s2 
        self.preprocess = preprocess
        self.augmentation = augmentation
        self.mode = mode
        self.dynamic_mixing_prob = dynamic_mixing_prob
        self.mouths_dir = Path(mouths_dir) if mouths_dir is not None else Path(self.audio_dir.parent.parent / "mouths") 

        self.data = []
        self.sources = []

        if mode == "test":
            for file_name in sorted((self.audio_dir / "mix").iterdir()):
                id1, id2 = self._parse_ids(file_name)
                path1, path2 = self.mouths_dir / f"{id1}.npz", self.mouths_dir / f"{id2}.npz"
                self.data.append({"mix": file_name, "s1_mouth": path1, "s2_mouth": path2})

        if mode == "train" or mode == "validation":
            for file_name in sorted((self.audio_dir / "mix").iterdir()):
                id1, id2 = self._parse_ids(file_name)
                path1, path2 = self.mouths_dir / f"{id1}.npz", self.mouths_dir / f"{id2}.npz"
                target_file_1 = self.audio_dir / "s1" / file_name.name
                target_file_2 = self.audio_dir / "s2" / file_name.name
                self.data.append({"mix": file_name, "s1": target_file_1, "s2": target_file_2, "s1_mouth": path1, "s2_mouth": path2})
                if mode == "train":
                    self.sources.append({"audio": target_file_1, "mouth": path1})
                    self.sources.append({"audio": target_file_2, "mouth": path2})

    def _parse_ids(self, mix_path):
        name = mix_path.stem
        id1, id2 = name.split("_")
        return id1, id2

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

    def _load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)
        if self.preprocess:
            waveform, sample_rate = self.preprocess(waveform, sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        return waveform, sample_rate

    def __getitem__(self, idx):
        if self.mode == "train" and random.random() < self.dynamic_mixing_prob:
            return self.make_dynamic_sample()

        item = self.data[idx]
        waveform, sample_rate = self._load_audio(item["mix"])

        if self.mode == "train" or self.mode == "validation":
            target_wf_1, _ = self._load_audio(item["s1"])
            target_wf_2, _ = self._load_audio(item["s2"])

            sample = {"mix": waveform, "s1": target_wf_1, "s2": target_wf_2, "sample_rate": sample_rate, "s1_mouth": item["s1_mouth"], "s2_mouth": item["s2_mouth"]}
            if self.mode == "train" and self.augmentation is not None:
                sample = self.augmentation(sample)

            return sample

        return {"mix": waveform, "sample_rate": sample_rate, "s1_mouth": item["s1_mouth"], "s2_mouth": item["s2_mouth"]}

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