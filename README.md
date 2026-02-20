# FashionMNIST CNN Benchmark (PyTorch): AlexNet vs GoogLeNet vs ResNet

## Overview
This project compares three well-known CNN architectures—**AlexNet**, **GoogLeNet (Inception)**, and **ResNet**—on the **FashionMNIST** dataset using **PyTorch**.  
Each model is adapted for **1-channel (grayscale) 28×28 images** and **10 classes**, then trained under the same setup to ensure a fair comparison.

## Models Implemented
- **AlexNet** (adapted for FashionMNIST input + 10-class output)
- **GoogLeNet** with custom **Inception** modules (4-branch concatenation)
- **ResNet-style** network using custom **residual blocks** with projection shortcuts

## Training Setup
- **Dataset:** FashionMNIST (torchvision)
- **Split:** 55,000 train / 5,000 validation + official test set
- **Batch size:** 32
- **Optimizer:** AdamW (lr = 1e-3)
- **Loss:** CrossEntropyLoss
- **Epochs:** 20
- **Metric:** Accuracy (torchmetrics multiclass)

## Results (Accuracy)
| Model     | Train Acc | Val Acc | Test Acc |
|----------|----------:|--------:|---------:|
| AlexNet  | 0.9294    | 0.9015  | 0.9004   |
| GoogLeNet| 0.9616    | 0.9122  | 0.9144   |
| ResNet   | 0.9843    | 0.9170  | 0.9122   |

## Goal
Demonstrate the trade-offs between classic CNN designs in terms of **accuracy and generalization** on a standard computer vision benchmark, using a consistent PyTorch pipeline.

## Tech Stack
- Python
- PyTorch, torchvision
- torchmetrics
- matplotlib, NumPy

## How to Run
1. Install dependencies:
   ```bash
   pip install torch torchvision torchmetrics matplotlib numpy
