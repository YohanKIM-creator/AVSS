from src.datasets.custom_dataset import AudioDataset

dataset = AudioDataset(audio_dir="dla_dataset_preview/audio/train", augmentation=None, preprocess=None, mode="train")
print(dataset[0])