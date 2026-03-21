# Lung Segmentation from Chest X-Rays using Deep Learning

A deep learning system for automatic lung segmentation from chest X-ray images, using a U-Net architecture with a ResNet34 encoder pretrained on ImageNet. Achieves a **Dice Score of 96.34%** on the validation set.



<img width="1397" height="2365" alt="Image" src="https://github.com/user-attachments/assets/c1636fc3-96d5-4fa2-8bc5-96b06ca74dd9" />
---

## Results

| Metric | Value |
|---|---|
| Training Dice Score | 96.76% |
| Validation Dice Score | 96.34% |
| Validation Loss (Dice + BCE) | 0.1144 |

<img width="1189" height="490" alt="Image" src="https://github.com/user-attachments/assets/c61d02c3-e7bb-42de-bd94-86694283d86d" />

A Dice Score above 96% on medical imaging is considered strong performance for lung segmentation the model reliably delineates both lungs even under varying patient anatomy and X-ray exposure conditions.

---

## Live Demo

Try the model yourself: **[Live App →](https://lung-segmentation-unet.streamlit.app/)**

Upload a chest X-ray and get instant lung segmentation results.

---

## Model Architecture

### Why Segmentation and Not Classification?

Lung segmentation is a **pixel-level prediction task** for each pixel in the image, the model must decide whether it belongs to the lung region or not. This is fundamentally different from image classification (which outputs a single label per image). A segmentation model produces a **binary mask** of the same spatial dimensions as the input, making it ideal for downstream tasks like disease localization, volume estimation, and radiological analysis.

---

### Why U-Net?

U-Net is the gold-standard architecture for medical image segmentation, and for good reason:

**1. Encoder-Decoder with Skip Connections**

U-Net follows an hourglass structure. The encoder progressively downsamples the image to capture high-level semantic features ("this region is lung tissue"). The decoder then upsamples back to the original resolution. The key innovation is **skip connections** the decoder receives feature maps directly from the corresponding encoder stage. This means:

- High-level understanding (what) from the bottleneck
- Precise spatial location (where) from the encoder shortcuts
- Result: sharp, accurate boundary segmentation

Without skip connections, the decoder struggles to recover fine spatial details lost during downsampling a critical failure for medical masks where boundary accuracy matters.

**2. Designed for Small Medical Datasets**

U-Net was originally proposed for biomedical segmentation and is specifically optimized to work well with limited labeled data a common constraint in medical imaging. Its parameter efficiency and data augmentation compatibility make it ideal here.

**3. Proven Track Record in Chest X-Ray Tasks**

U-Net and its variants dominate medical segmentation benchmarks. For lung segmentation specifically, it remains competitive with far more complex architectures while being faster to train and easier to interpret.

---

### Why ResNet34 as the Encoder?

The encoder is the "backbone" of U-Net. It extracts hierarchical features from the image. Instead of training a random encoder from scratch, we use ResNet34 **pretrained on ImageNet** via the `segmentation_models_pytorch` library. Here's the full reasoning:

#### ResNet34 vs. ResNet18 (Why Go Deeper?)

| Property | ResNet18 | ResNet34 |
|---|---|---|
| Layers | 18 | 34 |
| Parameters | ~11M | ~21M |
| Feature Richness | Basic edges, textures | Richer semantic patterns |
| Suitable For | Simple, high-contrast images | Complex anatomical structures |

ResNet18 works well for simple binary classification (as in the defect detection task). But lung segmentation requires recognizing **subtle anatomical boundaries**, rib cage interference, and variations in lung density across patients. ResNet34's additional depth gives it the representational capacity to capture these more complex spatial patterns, leading to better encoder features for the decoder to work with.

#### ResNet34 vs. ResNet50/ResNet101 (Why Not Go Deeper?)

| Property | ResNet34 | ResNet50 | ResNet101 |
|---|---|---|---|
| Architecture | Basic Blocks | Bottleneck Blocks | Bottleneck Blocks |
| Parameters | ~21M | ~25M | ~44M |
| Training Speed | Fast | Moderate | Slow |
| Risk of Overfitting | Low | Moderate | High (small datasets) |

ResNet50 and above use **bottleneck blocks** more complex but also more prone to overfitting on small-to-medium datasets like this one. Since our dataset size is moderate and the task is binary segmentation (not multi-class), the additional capacity of ResNet50+ does not justify the trade-off in training cost and overfitting risk.

**ResNet34 hits the sweet spot**: deep enough to extract rich features, lightweight enough to train efficiently and generalize well.

#### Why Pretrained on ImageNet (Transfer Learning)?

Chest X-rays are grayscale and visually very different from natural images. So why use ImageNet weights?

- The **early layers** of ResNet (edge detectors, texture detectors) transfer well to any image domain including medical images
- Pretraining gives the encoder a massive head start, requiring far fewer labeled X-rays to converge
- Training a ResNet34 encoder from scratch on this dataset would require much more data and time, with worse final performance

The images are converted from grayscale to 3-channel RGB to match the pretrained encoder's input format a standard and well-validated practice.

---

### Architecture Summary

```
Input: 256×256×3 (grayscale CXR converted to RGB)
│
├── ResNet34 Encoder (pretrained on ImageNet)
│   ├── Stage 1: 64 channels  ──────────────────┐ skip
│   ├── Stage 2: 128 channels ──────────────────┤ skip
│   ├── Stage 3: 256 channels ──────────────────┤ skip
│   └── Stage 4: 512 channels ──────────────────┤ skip
│                                                │
├── Bottleneck (512 channels)                    │
│                                                │
├── U-Net Decoder (with skip connections) ◄──────┘
│   ├── Upsample + Concat + Conv (512→256)
│   ├── Upsample + Concat + Conv (256→128)
│   ├── Upsample + Concat + Conv (128→64)
│   └── Upsample + Concat + Conv (64→32)
│
└── Output Head: Conv(32→1) + Sigmoid → Binary Mask (256×256×1)
```

---

## Loss Function: Why Dice + BCE (Combined Loss)?

### The Problem with Standard BCE Alone

Binary Cross-Entropy (BCE) measures pixel-wise error. In segmentation, **background pixels vastly outnumber foreground pixels** (the lung regions occupy only a fraction of the total image). A naive BCE loss can be dominated by correct background predictions, causing the model to learn a biased solution that "ignores" the lung region.

### The Problem with Dice Alone

Dice Loss measures overlap between predicted and ground-truth masks. It is naturally robust to class imbalance. However, Dice Loss has a **flat gradient near zero** when predictions are very wrong early in training, gradients can vanish, making convergence slow and unstable.

### Why the Combination Wins

```python
Total Loss = BCE Loss + Dice Loss
```

| Property | BCE | Dice | Combined |
|---|---|---|---|
| Handles class imbalance | ✗ | ✓ | ✓ |
| Stable early training | ✓ | ✗ | ✓ |
| Optimizes overlap directly | ✗ | ✓ | ✓ |
| Industry adoption | Standard | Medical standard | Best practice |

The combined loss gets the best of both: BCE stabilizes early training gradients, while Dice Loss pushes the model to maximize mask overlap which is exactly what the Dice Score metric measures. This alignment between loss function and evaluation metric is a deliberate and important design choice.

---

## Training Configuration

| Component | Choice | Reasoning |
|---|---|---|
| Optimizer | Adam (lr=1e-4) | Adaptive learning rates, robust to hyperparameter choice |
| Loss | DiceBCELoss | Handles imbalance + stable convergence |
| Input Size | 256×256 | Balance between detail preservation and GPU memory |
| Batch Size | 8 | Stable gradient estimates within memory constraints |
| Early Stopping | Patience=5 | Prevents overfitting, saves best checkpoint by validation loss |
| Epochs | 20 (max) | Sufficient with early stopping |
| Normalization | [0,1] range | Consistent with grayscale X-ray intensity distribution |

### Why Adam at lr=1e-4?

Adam is the standard optimizer for encoder-decoder networks with pretrained backbones. A learning rate of `1e-4` (lower than typical `1e-3`) is intentional — the encoder weights are already well-trained (ImageNet), so aggressive updates would destroy the pretrained features. A smaller learning rate allows **fine-tuning** rather than overwriting.

---

## Data Augmentation

```python
train_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Rotate(limit=35, p=0.5),       # Anatomical variation in patient positioning
    A.HorizontalFlip(p=0.5),          # Symmetric lungs appear on both sides
    A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
])
```

Augmentations were chosen to reflect **real-world clinical variation**:
- **Rotation**: Patients are not always perfectly aligned in X-ray captures
- **Horizontal Flip**: Anatomically valid — both lung orientations are physiologically meaningful
- No brightness/contrast jitter was applied to avoid distorting clinically significant intensity differences in X-ray images

---

## Dataset

**Source:** [Chest X-Ray Masks and Labels (Kaggle)](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)

- **Format:** PNG chest X-ray images + binary lung segmentation masks
- **Split:** 80% training / 20% validation (`train_test_split`, `random_state=42`)
- **Preprocessing:** Grayscale → RGB conversion, resized to 256×256
- **Mask format:** Binary (0 = background, 1 = lung)

---

## Installation

```bash
git clone https://github.com/Pooja-Vachhad/lung-segmentation-unet
cd lung-segmentation-unet
pip install -r requirements.txt
```

---

## Usage

### Training

1. Update the dataset path in `train.py`:

```python
path = "path/to/your/dataset"
```

2. Run training:

```bash
python train.py
```

The best model is saved automatically to `best.pth` based on validation loss. Training curves are saved to `training_curves.png`.

### Inference

1. Update the test folder path in `test.py`
2. Run predictions:

```bash
python test.py
```

---

## Project Structure

```
lung-segmentation-unet/
├── train.py              # Training script with early stopping
├── test.py               # Inference and visualization script
├── requirements.txt      # All dependencies
└── README.md             # This file
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- segmentation-models-pytorch
- albumentations
- OpenCV
- scikit-learn

See `requirements.txt` for the full list.

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch |
| Architecture | U-Net (segmentation-models-pytorch) |
| Encoder | ResNet34 (ImageNet pretrained) |
| Loss Function | Dice + BCE (Combined) |
| Optimizer | Adam (lr=1e-4) |
| Augmentation | Albumentations |
| Evaluation | Dice Score |

---

## License

MIT License
