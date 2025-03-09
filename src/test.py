import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# ===========================
# UTILITY: Letterbox Resize
# ===========================
def letterbox_resize(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding."""
    shape = img.shape[:2]  # (height, width)
    # Compute scaling ratio
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))  # (width, height)
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    pad_w = new_shape[1] - new_unpad[0]
    pad_h = new_shape[0] - new_unpad[1]
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded

# ===========================
# SEGMENTATION FUNCTIONS
# ===========================
def load_model(model_path, device='cpu'):
    """Loads a YOLO v11 model from file."""
    model = torch.load(model_path, map_location=torch.device(device))
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    model.eval()
    return model

def preprocess_image(img_path, input_size=(640, 640), rotate=False):
    """Reads an image, applies letterbox resize, optionally rotates it 180°."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_path}")
        return None, None
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_180)
        print(f"Rotated image {os.path.basename(img_path)} by 180°")
    img_padded = letterbox_resize(img, new_shape=input_size)
    original_shape = img.shape[:2]
    img_tensor = torch.from_numpy(img_padded).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, H, W]
    return img_tensor, original_shape

def get_segmentation_mask(model, img_tensor, device='cpu', threshold=0.2):
    """Runs inference on a single image and returns the binary segmentation mask.
       Also prints raw min and max values for debugging."""
    with torch.no_grad():
        output = model(img_tensor.to(device).half())
        if isinstance(output, tuple):
            print("Model returned a tuple with length:", len(output))
            output = output[0]
        else:
            print("Model output type:", type(output))
        if isinstance(output, dict):
            mask = output.get('mask', None)
            if mask is None:
                mask = output.get('pred', None)
        else:
            mask = output
        if isinstance(mask, tuple):
            print("Mask is a tuple with length:", len(mask))
            mask = mask[0]
        mask_tensor = mask.squeeze(0).squeeze(0)
        raw_min = mask_tensor.min().item()
        raw_max = mask_tensor.max().item()
        print(f"Raw mask values: min={raw_min}, max={raw_max}")
        mask_np = mask_tensor.cpu().numpy()
        binary_mask = (mask_np > threshold).astype(np.uint8)
    return binary_mask

def ensemble_segmentation(models, img_tensor, device='cpu', threshold=0.2):
    """Averages masks from multiple models and thresholds the average mask."""
    masks = []
    for model in models:
        mask = get_segmentation_mask(model, img_tensor, device, threshold)
        masks.append(mask)
    avg_mask = np.mean(masks, axis=0)
    binary_mask = (avg_mask > 0.5).astype(np.uint8)
    return binary_mask

# ===========================
# VOLUME CALCULATION ON A FOLDER (SINGLE SIDE)
# ===========================
def process_folder_side(folder_path, side, models, device='cpu', input_size=(640, 640), verbose=True):
    """
    Processes all images in one scan side of a folder.
    If side=='r', images are rotated 180° before segmentation.
    Returns aggregated_area, start_layer, and end_layer.
    """
    # Create a pattern that matches either uppercase or lowercase for the side letter.
    if side.lower() == 'l':
        pattern = os.path.join(folder_path, "*_[lL]_*.png")
    else:
        pattern = os.path.join(folder_path, "*_[rR]_*.png")
    
    image_files = sorted(glob.glob(pattern))
    if len(image_files) == 0:
        if verbose:
            print(f"No images found for side {side} in {folder_path}. Using default features.")
        return 0, 0, 0

    aggregated_area = 0
    layers_with_area = []
    for img_path in image_files:
        base = os.path.basename(img_path)
        try:
            layer_str = base.split('_')[-1].split('.')[0]
            layer = int(layer_str)
        except Exception:
            layer = 0
        img_tensor, orig_shape = preprocess_image(img_path, input_size, rotate=(side.lower()=='r'))
        if img_tensor is None:
            continue
        seg_mask = ensemble_segmentation(models, img_tensor, device, threshold=0.2)
        area = np.sum(seg_mask)
        aggregated_area += area
        if area > 0:
            layers_with_area.append(layer)
        if verbose:
            print(f"File: {base} | Orig shape: {orig_shape} | Layer: {layer} | Area: {area}")
            # Optionally, show one example mask:
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.imshow(cv2.resize(seg_mask*255, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST), cmap='gray')
            plt.title("Segmentation Mask")
            plt.axis("off")
            plt.subplot(1,2,2)
            plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis("off")
            plt.show()
    start_layer = min(layers_with_area) if layers_with_area else 0
    end_layer = max(layers_with_area) if layers_with_area else 0
    return aggregated_area, start_layer, end_layer


# ===========================
# SIMPLE VOLUME ESTIMATION
# ===========================
def estimate_volume(folder_path, side, models, device='cpu', input_size=(640,640), layer_thickness=1, verbose=True):
    """
    Estimates root volume from segmentation features.
    A simple approach: volume = aggregated_area * layer_thickness
    (If you have a more complex integration over layers, add that here.)
    """
    aggregated_area, start_layer, end_layer = process_folder_side(folder_path, side, models, device, input_size, verbose)
    # For simplicity, assume each layer corresponds to a constant thickness.
    # Here, volume is simply aggregated_area * layer_thickness.
    volume = aggregated_area * layer_thickness
    if verbose:
        print(f"Estimated volume for folder {os.path.basename(folder_path)}, side {side}: {volume}")
    return volume

# ===========================
# MAIN SCRIPT: Test Segmentation and Volume Calculation
# ===========================
if __name__ == "__main__":
    # Set a sample folder path (update with a real folder from your dataset)
    # For example, a folder from the training set:
    sample_folder = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/train/l5l1h3kekg"
    # Test for left side (usually "l") and right side ("r")
    side = "l"  # change to "r" to test right side
    
    # List of YOLO model paths (update with your actual model paths)
    model_paths = [
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_early.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_late.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_full.pt"
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = (640, 640)
    
    # Load YOLO models
    models = []
    for mp in model_paths:
        print(f"Loading model from {mp}")
        model = load_model(mp, device)
        models.append(model)
    print("All models loaded.")

    # Test segmentation on a single folder and side
    vol = estimate_volume(sample_folder, side, models, device, input_size, layer_thickness=1, verbose=True)
    print(f"Calculated volume: {vol}")
