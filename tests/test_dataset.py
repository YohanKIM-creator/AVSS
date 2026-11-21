from src.datasets.custom_dataset import AudioDataset

dataset = AudioDataset(audio_dir="dla_dataset_preview/audio/test", mode="test")

print("mouths_dir:", dataset.mouths_dir)
print("len(data):", len(dataset))
print("data[0]:", dataset.data[0])

sample = dataset[0]
print("shapes:", {k: v.shape for k, v in sample.items() if hasattr(v, "shape")})