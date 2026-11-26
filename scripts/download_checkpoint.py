import argparse
from pathlib import Path
import gdown

CHECKPOINT_ID = "16yHS6Vv2nMj9jaA0klaxsmLZSIRHl_1_"
LIPREADING_ID = "1vqMpxZ5LzJjg50HlZdj_QFJGm2gQmDUD"

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="saved/final/model_best.pth")
parser.add_argument("--lip_out", type=str, default="saved/lipreading/lrw_resnet18_mstcn_video.pth")
args = parser.parse_args()
for file_id, out in [(CHECKPOINT_ID, args.out), (LIPREADING_ID, args.lip_out)]:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=out, quiet=True)