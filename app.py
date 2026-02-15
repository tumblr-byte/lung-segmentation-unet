import streamlit as st
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from PIL import Image
import os
import requests
from pathlib import Path

st.set_page_config(page_title="Lung Segmentation", page_icon="🫁", layout="wide")

st.title("🫁 Lung Segmentation from Chest X-Ray")
st.markdown("Upload a chest X-ray image and get the lung segmentation mask!")

# GitHub Release Configuration
GITHUB_REPO = "tumblr-byte/lung-segmentation-unet"  
MODEL_VERSION = "v1.0.0"  
MODEL_FILENAME = "best.pth"
MODEL_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_VERSION}/{MODEL_FILENAME}"

def download_model(url, save_path):
    """Download model from GitHub releases"""
    if os.path.exists(save_path):
        st.info("Model already exists locally.")
        return save_path
    
    try:
        st.info(f"Downloading model from GitHub releases ({MODEL_VERSION})...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(downloaded / total_size)
        
        progress_bar.empty()
        
        # Verify file was downloaded completely
        if total_size > 0 and downloaded != total_size:
            os.remove(save_path)
            st.error(f"Download incomplete! Expected {total_size} bytes, got {downloaded} bytes.")
            st.stop()
        
        st.success("✅ Model downloaded successfully!")
        return save_path
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download model: {str(e)}")
        st.error("Please check:\n1. GitHub repo and release version are correct\n2. Model file exists in the release\n3. Internet connection is stable")
        st.stop()
    except Exception as e:
        st.error(f"Error during download: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        st.stop()

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Download model if not exists
        model_path = Path(MODEL_FILENAME)
        if not model_path.exists():
            download_model(MODEL_URL, MODEL_FILENAME)
        
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
        
        # Load model with error handling
        try:
            state_dict = torch.load(MODEL_FILENAME, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.error("The model file might be corrupted. Deleting and trying to re-download...")
            if os.path.exists(MODEL_FILENAME):
                os.remove(MODEL_FILENAME)
            st.stop()
        
        model = model.to(device)
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.stop()

def get_transforms():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

def predict_mask(image, model, device, transforms):
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    augmented = transforms(image=image_np)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()
    
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    return image_np, pred_mask_np

model, device = load_model()

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    if st.button("Generate Segmentation"):
        with st.spinner("Processing..."):
            transforms = get_transforms()
            image_np, pred_mask = predict_mask(image, model, device, transforms)
            
            with col2:
                st.subheader("Predicted Mask")
                st.image(pred_mask, use_container_width=True, clamp=True)
            
            st.success("Segmentation complete!")
