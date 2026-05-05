# CIFAR-100 Architecture Benchmark

Comparative performance analysis of three deep learning architectures for image
classification on CIFAR-100 (100 classes, 60,000 images).

## Models

| Model | Params | Test Accuracy |
|-------|--------|---------------|
| ResNet50 | 23.7M | 82.22% |
| ViT-B/16 | 85.9M | 87.73% |
| Swin Transformer Tiny | 27.6M | 87.23% |

## Key Finding

Swin Transformer matches ViT accuracy within 0.5% while using 3× fewer parameters
and training 2.6× faster — making it the most efficient architecture for this task.

## Setup

```bash
pip install torch torchvision timm scikit-learn seaborn pandas matplotlib
```

## Usage

```python
python cifar100_benchmark.py
```

CIFAR-100 downloads automatically on first run.

## Outputs

- Training curves (loss + accuracy) per model
- Confusion matrices per model
- Comparison bar charts across all metrics
- `comparison_table.csv`

## Dataset

CIFAR-100 — 60,000 images across 100 classes, split into 40k train / 10k val / 10k test.
All models use ImageNet pre-trained weights via `timm`.
