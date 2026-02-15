# Lung Segmentation Using U-Net

Deep learning model for automatic lung segmentation from chest X-ray images using U-Net architecture with ResNet34 encoder.

## Results

- **Validation Dice Score:** 96.34%
- **Training Dice Score:** 96.76%
- **Validation Loss:** 0.1144

<img width="1189" height="490" alt="Image" src="https://github.com/user-attachments/assets/c61d02c3-e7bb-42de-bd94-86694283d86d" />

<img width="1397" height="2365" alt="Image" src="https://github.com/user-attachments/assets/c1636fc3-96d5-4fa2-8bc5-96b06ca74dd9" />

## Live Demo

**[Try the live app here!](https://lung-segmentation-unet.streamlit.app/)**

## Model Architecture

- **Architecture:** U-Net
- **Encoder:** ResNet34 (pretrained on ImageNet)
- **Loss Function:** Dice + BCE Loss
- **Optimizer:** Adam (lr=1e-4)
- **Input Size:** 256x256

## Dataset

Dataset: [Chest X-Ray Masks and Labels](https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels)

The dataset contains chest X-ray images with corresponding lung segmentation masks.

## Installation

```bash
pip install -r requirements.txt
```

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

### Inference

1. Update test folder path in `test.py`
2. Run predictions:
```bash
python test.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

## Project Structure

```
lung-segmentation-unet/
├── train.py              # Training script
├── test.py               # Inference script
├── requirements.txt      # Dependencies
├── README.md            # This file

```

## License

MIT License
