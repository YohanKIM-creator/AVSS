# Audio-Visual Speech Separation with RoPE-Based SepReformer

Audio-visual speech separation (AVSS) system based on an asymmetric **SepReformer-T** architecture with **Rotary Position Embeddings (RoPE)** in Efficient Global Attention (EGA) and sliding-window local attention, conditioned on **ResNet-18 + MS-TCN** lip embeddings via a zero-initialized projection across all `Speaker Split` levels.

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

```bash
git clone https://github.com/YohanKIM-creator/AVSS.git
cd AVSS
git clone https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks.git ../lipreading
pip install -r requirements.txt
```

## Download Checkpoints

Download the final AV-SepReformer checkpoint (`saved/final/model_best.pth`) and pretrained lip-reading weights (`saved/lipreading/lrw_resnet18_mstcn_video.pth`):

```bash
python3 scripts/download_checkpoint.py
```

## Dataset & Lip Embeddings

### Dataset Structure

For training/validation (`AudioDataset` in [src/datasets/custom_dataset.py](src/datasets/custom_dataset.py)):

```text
dla_dataset_audio/
├── train/{mix,s1,s2}/
├── val/{mix,s1,s2}/
└── test/mix/
```

with speaker `.npz` mouth crops in `mouths/` and extracted `.npy` embeddings in `lip_emb/`.

For custom inference (`CustomDirDataset` in [src/configs/datasets/custom_dir.yaml](src/configs/datasets/custom_dir.yaml)):

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

```bash
python3 scripts/extract_lip_embeddings.py \
  --lip-repo ../lipreading \
  --weights saved/lipreading/lrw_resnet18_mstcn_video.pth \
  --mouths-dir ./mouths \
  --out-dir ./lip_emb \
  --batch-size 32
```

## Inference & Metrics

Run inference on any directory matching `CustomDirDataset` (lip embeddings are extracted automatically into `${inferencer.save_path}/lip_emb`):

```bash
python3 inference.py \
  datasets.test.data_dir=/path/to/CustomDir \
  inferencer.lip_repo=../lipreading \
  inferencer.weights=saved/lipreading/lrw_resnet18_mstcn_video.pth \
  inferencer.from_pretrained=saved/final/model_best.pth \
  inferencer.save_path=saved/inference
```

Separated `.wav` files are saved to `saved/inference/s1/<mix_name>.wav` and `saved/inference/s2/<mix_name>.wav`.

To compute **SI-SNRi**, **SDRi**, **PESQ**, and **STOI** against ground-truth `audio/{s1,s2}`:

```bash
python3 -m scripts.calc_metrics \
  --pred-dir saved/inference \
  --data-dir /path/to/CustomDir
```

### Google Colab Notebook

Use [notebooks/demo_colab.ipynb](notebooks/demo_colab.ipynb) to run the full inference and evaluation pipeline in Google Colab (clone the repository, install dependencies, download checkpoints, provide `DATA_DIR` via `.zip` upload or Google Drive, run `inference.py`, compute metrics with `scripts.calc_metrics`, and listen to separated audio).

## Training & Reproducibility

Training logs and audio samples are logged to the [Comet ML project](https://www.comet.com/143xd/avss/view/new/panels). Override `datasets.train.audio_dir`, `datasets.val.audio_dir`, `mouths_dir`, and `emb_dir` via Hydra if your data is stored outside the default paths in [src/configs/datasets/datasets.yaml](src/configs/datasets/datasets.yaml).

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
