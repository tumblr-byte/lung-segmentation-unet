import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import segmentation_models_pytorch as smp


# Test image transforms
test_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
)

model.load_state_dict(torch.load('best.pth', map_location=device))
model = model.to(device)
model.eval()

# Update this to your test images folder
test_folder = 'path/to/test/images'
test_images = [os.path.join(test_folder, img) for img in os.listdir(test_folder) if img.endswith('.png')]

# Generate predictions
fig, axes = plt.subplots(10, 2, figsize=(10, 50))

for idx in range(min(10, len(test_images))):
    original_image = cv2.imread(test_images[idx])
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    augmented = test_transforms(image=original_image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy()

    axes[idx, 0].imshow(original_image)
    axes[idx, 0].set_title(f'Original Image {idx+1}')
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(pred_mask_np, cmap='gray')
    axes[idx, 1].set_title(f'Predicted Mask {idx+1}')
    axes[idx, 1].axis('off')

plt.tight_layout()
plt.savefig("predictions.png", dpi=300, bbox_inches='tight')
plt.show()
