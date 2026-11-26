import argparse
from pathlib import Path
import soundfile as sf
import torch
from src.metrics.metrics import si_snr
from torchmetrics.functional.audio import signal_distortion_ratio, perceptual_evaluation_speech_quality, short_time_objective_intelligibility

def load(path):
    wf, _ = sf.read(path, dtype="float32")
    wf = torch.from_numpy(wf)
    return wf

parser = argparse.ArgumentParser()
parser.add_argument("--pred-dir", type=str, default="saved/inference")
parser.add_argument("--data-dir", required=True)
args = parser.parse_args()
pred_dir, data_dir = Path(args.pred_dir), Path(args.data_dir)
si_snrs = []
sdris = []
pesqs = []
stois = []
for wav in sorted((pred_dir / "s1").glob("*.wav")):
    name = wav.name
    est1 = pred_dir/ "s1" / name
    est2 = pred_dir/ "s2" / name
    ref1 = data_dir/ "audio" / "s1" / name
    ref2 = data_dir/ "audio" / "s2" / name
    mix = data_dir/ "audio" / "mix" / name
    rest1, rest2, rref1, rref2, rmix = load(est1), load(est2), load(ref1), load(ref2), load(mix)
    si_snri = (si_snr(rest1, rref1) + si_snr(rest2, rref2)) / 2 - (si_snr(rmix, rref1) + si_snr(rmix, rref2)) / 2
    sdri = (signal_distortion_ratio(rest1, rref1) + signal_distortion_ratio(rest2, rref2)) / 2 - (signal_distortion_ratio(rmix, rref1) + signal_distortion_ratio(rmix, rref2)) / 2
    pesq = (perceptual_evaluation_speech_quality(rest1, rref1, fs=16000, mode="wb") + perceptual_evaluation_speech_quality(rest2, rref2, fs=16000, mode="wb")) / 2
    stoi = (short_time_objective_intelligibility(rest1, rref1, fs=16000) + short_time_objective_intelligibility(rest2, rref2, fs=16000)) / 2
    si_snrs.append(si_snri.item())
    sdris.append(sdri.item())
    pesqs.append(pesq.item())
    stois.append(stoi.item())

print(f"Mean SI_SNRi = {sum(si_snrs) / len(si_snrs)}")
print(f"Mean SDRi = {sum(sdris) / len(sdris)}")
print(f"Mean PESQ = {sum(pesqs) / len(pesqs)}")
print(f"Mean STOI = {sum(stois) / len(stois)}")