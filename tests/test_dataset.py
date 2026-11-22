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

train_ds = AudioDataset(audio_dir="C:/Users/user/projects/dla_dataset/audio/train", mode="train", limit=100)
dyn = train_ds.make_dynamic_sample()
print("keys:", sorted(dyn.keys()))
print("shapes:", dyn["mix"].shape, dyn["s1"].shape, dyn["s2"].shape)
print("mix - s1 - s2:", (dyn["mix"] - dyn["s1"] - dyn["s2"]).abs().max().item())
print("max |mix|:", dyn["mix"].abs().max().item())
print("mouths:", dyn["s1_mouth"].name, dyn["s2_mouth"].name)