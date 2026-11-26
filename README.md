# Audio-Visual Speech Separation with RoPE-Based SepReformer

Our system adapts the asymmetric encoder-decoder **SepReformer-T** architecture with **Rotary Position Embeddings (RoPE)** in both Efficient Global Attention (EGA) and sliding-window local attention, and fuses pre-extracted **ResNet-18 + MS-TCN** lip embeddings across all U-Net `Speaker Split` levels via a zero-initialized projection.

## Results Summary

### Training Progression

| Stage / Configuration | Modality / Loss | Best Val SI-SNRi (dB) |
| --- | --- | :---: |
| `sepreformer_audio_nodm` (40 ep) | Audio / PIT | 7.82 |
| `sepreformer_audio_nodm_cont_lr1e3` (+40 ep) | Audio / PIT | 11.38 |
| `sepreformer_av_lr1e3` (+45 ep) | Audio-Visual / non-PIT | 13.58 |
| **`sepreformer_av_cont_lr25e5` (+20 ep)** | **Audio-Visual / non-PIT** | **14.01** |

### Final Model Evaluation (Full Validation Split)

| Model | SI-SNRi (dB) ↑ | SDRi (dB) ↑ | PESQ ↑ | STOI ↑ |
| --- | :---: | :---: | :---: | :---: |
| **AV-SepReformer (`sepreformer_av_cont_lr25e5`)** | **14.13** | **14.50** | **2.41** | **0.925** |

Model size: **3.32M** inference-active parameters (**3.44M** total including training-only auxiliary heads; **32,832** parameters in `video_proj`).

## Installation

1. Clone this repository and the pretrained lip-reading repository (placed next to it at `../lipreading` by default):

```bash
git clone https://github.com/YohanKIM-creator/AVSS.git
cd AVSS
git clone https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks.git ../lipreading
```

2. Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Checkpoints

Download both the final AV-SepReformer checkpoint (`saved/final/model_best.pth`) and the pretrained lip-reading backbone weights (`saved/lipreading/lrw_resnet18_mstcn_video.pth`):

```bash
python3 scripts/download_checkpoint.py
```

Optional custom output paths:

```bash
python3 scripts/download_checkpoint.py \
  --out saved/final/model_best.pth \
  --lip_out saved/lipreading/lrw_resnet18_mstcn_video.pth
```

## Dataset & Lip Embeddings Preparation

### Dataset Structure

For training/validation (`AudioDataset` in [src/datasets/custom_dataset.py](src/datasets/custom_dataset.py)), place the dataset as:

```text
dla_dataset_audio/
├── train/
│   ├── mix/
│   ├── s1/
│   └── s2/
├── val/
│   ├── mix/
│   ├── s1/
│   └── s2/
└── test/
    └── mix/
```

with corresponding speaker `.npz` mouth crops in `mouths/` and extracted `.npy` embeddings in `lip_emb/`.

For custom inference (`CustomDirDataset` configured in [src/configs/datasets/custom_dir.yaml](src/configs/datasets/custom_dir.yaml)):

```text
CustomDir/
├── audio/
│   ├── mix/          # FirstSpeakerID_SecondSpeakerID.wav
│   ├── s1/           # optional ground truth
│   └── s2/           # optional ground truth
└── mouths/
    └── SpeakerID.npz
```

### Extract Lip Embeddings (for Training)

Before running `train.py`, pre-extract FP16 512-d lip embeddings from `.npz` mouth crops:

```bash
python3 scripts/extract_lip_embeddings.py \
  --lip-repo ../lipreading \
  --weights saved/lipreading/lrw_resnet18_mstcn_video.pth \
  --mouths-dir ./mouths \
  --out-dir ./lip_emb \
  --batch-size 32
```

*(During `inference.py`, lip embeddings for `CustomDir/mouths` are extracted automatically into `${inferencer.save_path}/lip_emb`.)*

## Inference

Run inference on any directory matching the `CustomDirDataset` layout:

```bash
python3 inference.py \
  datasets.test.data_dir=/path/to/CustomDir \
  inferencer.lip_repo=../lipreading \
  inferencer.weights=saved/lipreading/lrw_resnet18_mstcn_video.pth \
  inferencer.from_pretrained=saved/final/model_best.pth \
  inferencer.save_path=saved/inference
```

Separated `.wav` predictions are saved with the same filename as each mixture into:
- `saved/inference/s1/<mix_name>.wav`
- `saved/inference/s2/<mix_name>.wav`

If `audio/s1` and `audio/s2` exist inside `data_dir`, `inference.py` also computes and prints `test_SISNRi`.

### Metrics Evaluation

To compute all required metrics (**SI-SNRi**, **SDRi**, **PESQ**, and **STOI**) between ground-truth `data_dir/audio/{s1,s2}` and predicted `pred_dir/{s1,s2}` utterances:

```bash
python3 -m scripts.calc_metrics \
  --pred-dir saved/inference \
  --data-dir /path/to/CustomDir
```

### Google Colab / Jupyter Notebook

Use [notebooks/demo_colab.ipynb](notebooks/demo_colab.ipynb) to reproduce the full inference and evaluation pipeline in Google Colab:
1. Clone `AVSS` (using Colab Secret `GITHUB_TOKEN` if the repo is private) and install `requirements.txt`.
2. Clone `Lipreading_using_Temporal_Convolutional_Networks` and download checkpoints via `scripts/download_checkpoint.py`.
3. Upload a `.zip` archive (containing `audio/` and `mouths/`) or mount Google Drive and set `DATA_DIR`.
4. Run `inference.py`, compute `SI-SNRi`, `SDRi`, `PESQ`, and `STOI` via `python -m scripts.calc_metrics`, and listen to separated `s1`/`s2` predictions.

## Training & Reproducibility

Experiments are logged with **Comet ML** (training logs and audio samples are available in the [Comet ML project](https://www.comet.com/143xd/avss/view/new/panels); make sure `COMET_API_KEY` is configured or set `writer.mode=offline`). Override `datasets.train.audio_dir`, `datasets.val.audio_dir`, `mouths_dir`, and `emb_dir` via Hydra if your data is stored outside the default paths in [src/configs/datasets/datasets.yaml](src/configs/datasets/datasets.yaml).

1. **Audio-only pretraining (40 epochs, PIT):**

```bash
python3 train.py -cn=baseline \
  model.use_video=False \
  loss_function.pit=True \
  metrics=separation \
  datasets.train.dynamic_mixing_prob=0.0 \
  dataloader.batch_size=4 \
  trainer.n_epochs=40 \
  optimizer.lr=3e-4 \
  writer.run_name=sepreformer_audio_nodm
```

2. **Audio-only continuation (+40 epochs, LR = 1e-3):**

```bash
python3 train.py -cn=baseline \
  model.use_video=False \
  loss_function.pit=True \
  metrics=separation \
  datasets.train.dynamic_mixing_prob=0.0 \
  dataloader.batch_size=4 \
  trainer.n_epochs=40 \
  optimizer.lr=1e-3 \
  trainer.from_pretrained=saved/sepreformer_audio_nodm/model_best.pth \
  writer.run_name=sepreformer_audio_nodm_cont_lr1e3
```

3. **Audio-visual fine-tuning (45 epochs at LR = 1e-3, then 20 epochs at LR = 2.5e-4, non-PIT):**

```bash
python3 train.py -cn=baseline \
  model.use_video=True \
  loss_function.pit=False \
  metrics=separation_av \
  datasets.train.dynamic_mixing_prob=0.0 \
  dataloader.batch_size=4 \
  trainer.n_epochs=45 \
  optimizer.lr=1e-3 \
  trainer.from_pretrained=saved/sepreformer_audio_nodm_cont_lr1e3/model_best.pth \
  writer.run_name=sepreformer_av_lr1e3

python3 train.py -cn=baseline \
  model.use_video=True \
  loss_function.pit=False \
  metrics=separation_av \
  datasets.train.dynamic_mixing_prob=0.0 \
  dataloader.batch_size=4 \
  trainer.n_epochs=20 \
  optimizer.lr=2.5e-4 \
  trainer.from_pretrained=saved/sepreformer_av_lr1e3/model_best.pth \
  writer.run_name=sepreformer_av_cont_lr25e5
```

## Generative AI Disclosure

- **Model:** Codex (`gpt sol 6.1`)
- **Tasks:** Exploratory analysis of the raw dataset archive (`dla_dataset.zip`) — counting files per split, checking WAV sizes/format uniformity, verifying linear mixture composition (`mix ≈ s1 + s2` residuals and fitted coefficients), measuring source RMS ratios, mixture SI-SNR, and clipping, inspecting `.npz` mouth-crop shapes/dtypes, saving summary statistics to `dla_dataset_preview/inspection.json`, and extracting sample `mix`/`s1`/`s2` `.wav` previews into `dla_dataset_preview/`.
- **Prompt (reconstructed):** *"Просмотри архив с датасетом, создай файл `inspection.json` со всевозможными статистиками о датасете (количество файлов по сплитам, метаданные и размеры аудио, проверка `mix = s1 + s2` через линейную регрессию и остаток, разброс RMS источников, SI-SNR, клиппинг, формы и типы `.npz` с губами), а также собери несколько аудио- и видео-сэмплов в `dla_dataset_preview`, чтобы я мог лучше понять, что там лежит."*
- **Code Generation:** No generative AI was used to write the model, dataset, training, or inference source code in `src/` and `scripts/` (the Colab inference notebook `notebooks/demo_colab.ipynb` is permitted by course rules).
