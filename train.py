import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp


# Update this path to your dataset location
path = "path/to/your/dataset"

path_dir = os.path.join(path, 'Lung Segmentation')
print(os.listdir(path_dir))

image_paths = sorted(glob(os.path.join(path_dir, 'CXR_png', '*.png')))
mask_paths = sorted(glob(os.path.join(path_dir, 'masks', '*.png')))

print(f"number of images are {len(image_paths)}")
print(f"number of mask images are {len(mask_paths)}")
print(f"example image_path {image_paths[0]}")
print(f"example of mask_path {mask_paths[0]}")

# Match images with their corresponding masks
mask_dict = {}
for mask_path in mask_paths:
    mask_filename = os.path.basename(mask_path)
    img_name = mask_filename.replace('_mask.png', '.png')
    mask_dict[img_name] = mask_path

images_pair = []
masks_pair = []

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    if img_name in mask_dict:
        images_pair.append(img_path)
        masks_pair.append(mask_dict[img_name])

print(f"successfully paired: {len(images_pair)} images with masks")

image_path = images_pair
mask_path = masks_pair


# Training augmentations
train_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Rotate(limit=35, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

# Validation transforms - no augmentation
valid_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])


class Custom(Dataset):
    def __init__(self, image_path, mask_path, transforms=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.transforms = transforms

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_file = self.image_path[idx]
        mask_file = self.mask_path[idx]

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_file, 0)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0).float()

        return image, mask


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        intersection = (outputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)

        outputs_sigmoid = torch.sigmoid(outputs)
        outputs_flat = outputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (outputs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (outputs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        return bce_loss + dice_loss


def calculate_dice_score(outputs, targets, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > threshold).float()
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    intersection = (outputs * targets).sum()
    dice = (2. * intersection) / (outputs.sum() + targets.sum() + 1e-8)

    return dice.item()


def run_model(model, device, optimizer, criterion, train_loader, valid_loader, patience=5, epochs=20, output_path="best.pth"):
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_dice": [],
        "valid_dice": []
    }

    best_loss = np.inf
    counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0

        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += calculate_dice_score(outputs, masks)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_dice"].append(avg_train_dice)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_dice = 0.0

        with torch.no_grad():
            for images, masks in tqdm(valid_loader):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)

                loss = criterion(outputs, masks)
                valid_loss += loss.item()
                valid_dice += calculate_dice_score(outputs, masks)

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_dice = valid_dice / len(valid_loader)
        history["valid_loss"].append(avg_valid_loss)
        history["valid_dice"].append(avg_valid_dice)

        print(f"Epoch: {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Valid Dice: {avg_valid_dice:.4f}")

        # Save best model
        if avg_valid_loss <= best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), output_path)
            print("Model saved")
            counter = 0
        else:
            print("No improvement")
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    # Plot training curves
    epoch_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, history["train_loss"], 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epoch_range, history["valid_loss"], 'r-o', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, history["train_dice"], 'b-o', label='Training Dice', linewidth=2)
    plt.plot(epoch_range, history["valid_dice"], 'r-o', label='Validation Dice', linewidth=2)
    plt.title('Training and Validation Dice Score', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Dice Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    return history


# Split data into train and validation
train_images, valid_images, train_masks, valid_masks = train_test_split(
    image_path, mask_path, test_size=0.2, random_state=42
)

train_dataset = Custom(train_images, train_masks, train_transforms)
valid_dataset = Custom(valid_images, valid_masks, valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=8, num_workers=2)

# Initialize U-Net model with ResNet34 encoder
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = DiceBCELoss()

# Train the model
history = run_model(model, device, optimizer, criterion, train_loader, valid_loader, 
                   patience=5, epochs=20, output_path="best.pth")
