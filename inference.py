import warnings

import hydra
import torch
from pathlib import Path
import subprocess
import sys
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    save_path = Path(config.inferencer.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    subprocess.run([sys.executable, "scripts/extract_lip_embeddings.py", "--lip-repo" ,str(config.inferencer.lip_repo), 
                    "--weights", str(config.inferencer.weights), "--mouths-dir", str(Path(config.datasets.test.data_dir) / "mouths"), "--out-dir", str(save_path / "lip_emb")], check=True)
    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # get metrics
    metrics = instantiate(config.metrics)


    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
