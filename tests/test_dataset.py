from src.datasets.custom_dataset import AudioDataset

dataset = AudioDataset(audio_dir="C:/Users/user/projects/dla_dataset/audio/val", mode="validation")

print("mouths_dir:", dataset.mouths_dir)
print("len(data):", len(dataset))
print("data[0]:", dataset.data[0])

sample = dataset[0]
print("shapes:", {k: v.shape for k, v in sample.items() if hasattr(v, "shape")})

from src.datasets.collate import collate_fn

items = [dataset[0], dataset[1]]
batch = collate_fn(items)