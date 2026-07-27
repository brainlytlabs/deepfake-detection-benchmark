"""DFDC face-crop dataset.

DFDC ships as video. The standard pipeline extracts frames, detects and
crops the face, and trains an image classifier on those crops. This loader
assumes preprocessing has already produced a directory of face crops:

    <root>/
        real/  *.jpg
        fake/  *.jpg

Use src/data/preprocess.py to generate that layout from raw videos.
"""

from pathlib import Path
from typing import List, Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png"}
LABELS = {"real": 0, "fake": 1}


class DFDCFaceDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for label_name, label_idx in LABELS.items():
            folder = self.root / label_name
            if not folder.exists():
                continue
            for path in folder.iterdir():
                if path.suffix.lower() in IMG_EXTS:
                    self.samples.append((path, label_idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. Expected "
                f"{self.root}/real/ and {self.root}/fake/ with image files. "
                f"Run preprocessing first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_balance(self) -> dict:
        counts = {"real": 0, "fake": 0}
        inv = {v: k for k, v in LABELS.items()}
        for _, label in self.samples:
            counts[inv[label]] += 1
        return counts
