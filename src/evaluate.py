"""Evaluation entry point.

    python -m src.evaluate --config configs/config.yaml --arch xception \
        --test-dir data/processed/celebdf_test

Loads a trained checkpoint, evaluates on a test directory, and writes
metrics plus ROC and confusion-matrix plots. Point --test-dir at a
DIFFERENT dataset than training to measure cross-dataset generalization —
the number that actually matters.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models.factory import build_model, input_size
from .data.dataset import DFDCFaceDataset
from .data.loaders import build_transforms
from .utils.metrics import compute_metrics, roc_points, confusion
from torch.utils.data import DataLoader


def load_model(arch, ckpt, device):
    model = build_model(arch, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


def run(cfg, arch, test_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = input_size(arch)

    ds = DFDCFaceDataset(test_dir, transform=build_transforms(size, train=False))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False)
    print(f"Test set: {len(ds)} crops  |  balance: {ds.class_balance()}")

    ckpt = Path(cfg["train"]["checkpoint_dir"]) / arch / "best.pt"
    model = load_model(arch, ckpt, device)

    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
            labels.extend(y.tolist())

    metrics = compute_metrics(labels, probs)
    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:10s} {v:.4f}")

    out = Path(cfg["eval"]["results_dir"]) / arch
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC curve
    fpr, tpr, _ = roc_points(labels, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"{arch} (AUC={metrics['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {arch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion(labels, probs)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["real", "fake"])
    plt.yticks([0, 1], ["real", "fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion — {arch}")
    plt.tight_layout()
    plt.savefig(out / "confusion.png", dpi=150)
    plt.close()

    print(f"\nSaved metrics and plots to {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--arch", required=True,
                    choices=["xception", "efficientnet", "resnet", "vit"])
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.arch, args.test_dir)


if __name__ == "__main__":
    main()
