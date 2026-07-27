"""Preprocess DFDC videos into face crops.

    python -m src.data.preprocess --videos data/raw/dfdc --out data/processed/dfdc \
        --frames-per-video 10

DFDC provides videos plus a metadata.json labelling each as REAL or FAKE.
This script samples frames, detects the largest face per frame with MTCNN,
and saves crops into real/ and fake/ subfolders.

Face detection uses facenet-pytorch (MTCNN). Install with:
    pip install facenet-pytorch opencv-python
"""

import argparse
import json
from pathlib import Path

# Imported lazily inside main() so the module imports without the heavy
# CV dependencies — keeps unit tests and CI light.


def process(videos_dir: str, out_dir: str, frames_per_video: int, margin: int):
    import cv2
    import numpy as np
    from facenet_pytorch import MTCNN
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = MTCNN(margin=margin, device=device, post_process=False)

    videos_dir = Path(videos_dir)
    out = Path(out_dir)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)

    meta_path = videos_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Expected DFDC metadata.json at {meta_path}. "
            "Each DFDC part ships one."
        )
    metadata = json.loads(meta_path.read_text())

    saved = {"real": 0, "fake": 0}
    for video_name, info in metadata.items():
        label = info.get("label", "").lower()  # 'real' or 'fake'
        if label not in ("real", "fake"):
            continue
        video_path = videos_dir / video_name
        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(total // frames_per_video, 1)

        idx, kept = 0, 0
        while kept < frames_per_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx * step)
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = detector(rgb)
            if face is not None:
                arr = face.permute(1, 2, 0).byte().cpu().numpy()
                stem = Path(video_name).stem
                cv2.imwrite(
                    str(out / label / f"{stem}_{kept}.jpg"),
                    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                )
                saved[label] += 1
                kept += 1
            idx += 1
        cap.release()

    print(f"Saved crops -> real: {saved['real']}, fake: {saved['fake']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="DFDC part dir with metadata.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-video", type=int, default=10)
    ap.add_argument("--margin", type=int, default=20)
    args = ap.parse_args()
    process(args.videos, args.out, args.frames_per_video, args.margin)


if __name__ == "__main__":
    main()
