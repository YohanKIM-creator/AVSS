import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Extract lip embeddings")
    parser.add_argument("--lip-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mouths-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def build_model(lip_repo, weights_path, device):
    sys.path.append(str(lip_repo))
    from lipreading.model import Lipreading
    model = Lipreading(
        modality="video",
        backbone_type="resnet",
        relu_type="swish",
        width_mult=1.0,
        num_classes=500,
        extract_feats=True,
        tcn_options={"num_layers": 4, "kernel_size": [3, 5, 7], "dropout": 0.2,
                     "dwpw": False, "width_mult": 1}
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    return model

def preprocess(frames):
    frames = frames.astype(np.float32)
    frames /= 255
    frames = frames[:, 4:92, 4:92]
    frames -= 0.421
    frames /= 0.165
    return frames

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.mouths_dir.glob("*.npz"))[:args.limit]
    files = [path for path in files if not (args.out_dir / f"{path.stem}.npy").exists()]
    model = build_model(args.lip_repo, args.weights, device)
    with torch.no_grad():
        for start in range(0, len(files), args.batch_size):
            batch_files = files[start:start + args.batch_size]
            batch = torch.from_numpy(np.stack([preprocess(np.load(path)["data"]) for path in batch_files]))
            batch = batch.unsqueeze(1).to(device)
            feats = model(batch, lengths=[batch.shape[2]] * batch.shape[0])
            feats = feats.cpu().numpy().astype(np.float16)
            for i, path in enumerate(batch_files):
                np.save(args.out_dir / f"{path.stem}.npy", feats[i])
        print(f"Saved {len(files)} embeddings to {args.out_dir}")
    

if __name__ == "__main__":
    main()