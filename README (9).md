<h1 align="center">🎭 DeepFake Detection Benchmark</h1>

<p align="center">
  <b>Comparative evaluation of CNN and Transformer architectures for DeepFake detection on DFDC</b><br>
  Xception · EfficientNet · ResNet · Vision Transformer
</p>

<p align="center">
  <i>A <b>Brainlyt Labs</b> research benchmark</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Brainlyt%20Labs-8E44FF?style=flat-square&logoColor=white">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/timm-1.0-8E44FF?style=flat-square">
  <img src="https://img.shields.io/badge/dataset-DFDC-00A8FF?style=flat-square">
</p>

---

## Overview

DeepFake detection is easy to overfit and hard to generalize. This benchmark holds the training pipeline constant and varies only the backbone, so four architectures can be compared fairly on the same data, augmentations, and evaluation protocol.

Four architectures are compared — two convolutional families, one residual baseline, and one attention-based model:

| Architecture | Family | Why it's here |
|---|---|---|
| **Xception** | Depthwise-separable CNN | The long-standing FaceForensics++ baseline for face forgery |
| **EfficientNet-B0** | Compound-scaled CNN | Strong accuracy per parameter; a common DFDC choice |
| **ResNet-50** | Residual CNN | Well-understood baseline for comparison |
| **ViT-Base/16** | Vision Transformer | Tests whether attention beats convolution on this task |

---

## Results

Trained on **6,000 face crops per class** from the DFDC train sample, up to 8 epochs with early stopping on validation AUC. Best checkpoint per architecture scored once on a held-out test split.

| Model | Accuracy | AUC | Log Loss | F1 | Train time |
|---|---|---|---|---|---|
| **ResNet-50** | **0.9772** | 0.9972 | **0.0664** | **0.9770** | 15.2 min |
| EfficientNet-B0 | 0.9744 | **0.9976** | 0.0759 | 0.9739 | **6.9 min** |
| Xception | 0.9711 | 0.9969 | 0.0755 | 0.9705 | 53.2 min |
| ViT-Base/16 | 0.7839 | 0.8707 | 0.4536 | 0.7672 | 46.1 min |

<p align="center">
  <img src="docs/figures/benchmark_comparison.png" alt="Benchmark comparison" width="820">
</p>

### What the numbers show

**The three CNNs are effectively tied.** ResNet-50, EfficientNet-B0, and Xception all land within a percentage point of each other on every metric — the differences are within run-to-run noise, so no single CNN is meaningfully "best" on accuracy alone.

**EfficientNet wins on efficiency.** It matches the others' accuracy while training in **7 minutes versus Xception's 53** — roughly one-eighth the compute for the same result. On this task, that makes it the practical choice.

**ViT underperforms, as expected.** Vision Transformers lack the inductive biases CNNs have for images and are notoriously data-hungry. On a capped 6,000-crops-per-class subset with a short training schedule, ViT has too little data to reach CNN-level performance. With a larger dataset and longer training it typically closes the gap — its weaker score here reflects the training budget, not an inherent ceiling.

> **Note on in-domain scores.** AUC near 0.99 on DFDC is partly the models learning DFDC's own compression and generation fingerprints rather than deepfake cues in general. These numbers reflect in-domain performance; cross-dataset generalization (training on DFDC, testing on an unseen manipulation method) is the harder and more meaningful test, and is the natural next step for this work.

---

## Pipeline

```
videos ──> face crops ──> train (per architecture) ──> evaluate ──> metrics + plots
```

Detection operates on cropped faces, not full frames — the manipulation artifacts live in the face region, and cropping removes background the model would otherwise memorize.

---

## Reproducing the results

The full experiment runs on a free Kaggle GPU. A ready-to-run notebook is included at [`notebooks/deepfake_benchmark_kaggle.ipynb`](notebooks/deepfake_benchmark_kaggle.ipynb).

1. Upload the notebook to Kaggle (**Create → New Notebook → File → Import Notebook**)
2. Settings → Accelerator → **GPU T4 x2**, and Internet → **On**
3. Add a DFDC face-crops dataset as input (with `real/` and `fake/` folders)
4. Set `DATA_DIR` to the dataset path and **Run All**
5. The final cell prints the results table

To run locally instead:

```bash
git clone https://github.com/brainlytlabs/deepfake-detection-benchmark.git
cd deepfake-detection-benchmark

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. preprocess videos -> face crops (skip if you have crops already)
python -m src.data.preprocess --videos data/raw/dfdc --out data/processed/dfdc

# 2. train an architecture
python -m src.train --config configs/config.yaml --arch efficientnet

# 3. evaluate
python -m src.evaluate --config configs/config.yaml --arch efficientnet \
    --test-dir data/processed/dfdc_test
```

### Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
deepfake-detection-benchmark/
├── configs/
│   └── config.yaml            # Training & evaluation settings
├── notebooks/
│   └── deepfake_benchmark_kaggle.ipynb   # One-click Kaggle reproduction
├── src/
│   ├── data/
│   │   ├── preprocess.py      # Video -> face crops (MTCNN)
│   │   ├── dataset.py         # Face-crop dataset
│   │   └── loaders.py         # Transforms & train/val split
│   ├── models/
│   │   └── factory.py         # Four architectures via timm
│   ├── utils/
│   │   ├── metrics.py         # AUC, log loss, ROC, confusion
│   │   └── seed.py            # Reproducibility
│   ├── train.py               # Training entry point
│   └── evaluate.py            # Evaluation entry point
├── docs/figures/              # Benchmark comparison chart
├── tests/
└── requirements.txt
```

---

## Design Notes

**Fair comparison.** Every architecture sees identical data splits, augmentation, optimizer, and early-stopping criteria. Only the backbone changes, so differences in the table are attributable to architecture, not pipeline luck.

**Pretrained backbones.** DFDC yields a limited set of distinct identities and manipulation methods. Training these backbones from scratch overfits; ImageNet-pretrained weights with a fine-tuned head generalize better and train far faster.

**Reproducibility.** All randomness is seeded and cuDNN is set deterministic, so a given config reproduces its numbers across runs.

---

## Roadmap

- [ ] **Cross-dataset evaluation** — train on DFDC, test on Celeb-DF / FaceForensics++ to measure true generalization
- [ ] Face-quality filtering to drop blurred / occluded crops
- [ ] Longer ViT schedule on the full dataset to test whether it closes the gap
- [ ] Temporal modeling across frames rather than per-frame classification
- [ ] Grad-CAM visualization of what each model attends to

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Humaira Muqades** — AI Engineer, PhD scholar researching DeepFake detection
Built at **Brainlyt Labs**.

[LinkedIn](https://www.linkedin.com/in/humaira-muqades-rana/) · [GitHub](https://github.com/Humaira-Muqades) · humaramuqdes@gmail.com

---

<p align="center">
  <b>Brainlyt Labs</b><br>
  <i>Engineering Intelligent AI Solutions</i><br>
  Think • Learn • Solve • Innovate
</p>
