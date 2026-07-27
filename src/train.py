"""Training entry point.

    python -m src.train --config configs/config.yaml --arch xception

Trains one architecture and saves the best checkpoint by validation AUC.
Run once per architecture to populate the benchmark.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from .models.factory import build_model, input_size
from .data.loaders import build_loaders
from .utils.metrics import compute_metrics
from .utils.seed import set_seed


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            probs.extend(p.cpu().tolist())
            labels.extend(y.tolist())
    return compute_metrics(labels, probs)


def train(cfg: dict, arch: str):
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Architecture: {arch}")

    size = input_size(arch)
    train_loader, val_loader = build_loaders(
        root=cfg["data"]["train_dir"],
        size=size,
        batch_size=cfg["train"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        num_workers=cfg["train"]["num_workers"],
        seed=cfg.get("seed", 42),
    )

    model = build_model(arch, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(cfg["train"]["checkpoint_dir"]) / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    best_auc, patience, bad_epochs = 0.0, cfg["train"]["patience"], 0
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        train_loss = running / len(train_loader.dataset)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        print(f"Epoch {epoch:02d}  loss={train_loss:.4f}  "
              f"val_auc={val['auc']:.4f}  val_acc={val['accuracy']:.4f}")

        if val["auc"] > best_auc:
            best_auc = val["auc"]
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best AUC {best_auc:.4f})")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val AUC {best_auc:.4f}. Checkpoint: {out_dir/'best.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--arch", required=True,
                    choices=["xception", "efficientnet", "resnet", "vit"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg, args.arch)


if __name__ == "__main__":
    main()
