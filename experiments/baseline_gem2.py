import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Model Loading
# -----------------------------
def load_model(model_path, device='cpu'):
    model = torch.load(model_path, map_location=torch.device(device))
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    model.eval()
    return model

# -----------------------------
# 2. Preprocess Image
# -----------------------------
def preprocess_image(img_path, input_size=(640, 640)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_path}")
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]  # (height, width)
    img_resized = cv2.resize(img, input_size)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor, original_shape

# -----------------------------
# 3. Run Inference on a Single Model
# -----------------------------
def get_segmentation_mask(model, img_tensor, device='cpu'):
    with torch.no_grad():
        # Convert input to FP16 (if model weights are in half precision)
        output = model(img_tensor.to(device).half())
        # Print detailed information about the output
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
        # Assume mask shape is [1, 1, H, W]
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)
    return binary_mask

# -----------------------------
# 4. Ensemble Segmentation
# -----------------------------
def ensemble_segmentation(models, img_tensor, device='cpu'):
    masks = []
    for model in models:
        mask = get_segmentation_mask(model, img_tensor, device)
        masks.append(mask)
    avg_mask = np.mean(masks, axis=0)
    binary_avg_mask = (avg_mask > 0.5).astype(np.uint8)
    return binary_avg_mask

# -----------------------------
# 5. Save Artifacts (instead of real-time display)
# -----------------------------
def save_artifact(orig_img, mask, area, layer, folder, side, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(orig_img)
    ax[0].set_title(f'Original Image\nFolder: {folder} | Side: {side} | Layer: {layer}')
    ax[0].axis('off')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title(f'Segmentation Mask\nDetected Area: {area}')
    ax[1].axis('off')
    filename = f"{folder}_{side}_layer{layer}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved artifact: {filepath}")

# -----------------------------
# 6. Process a Folder's Side (L or R) with Verbose Logging
# -----------------------------
def process_folder_side(folder_path, side, models, device='cpu', input_size=(640, 640), artifact_dir="artifacts", verbose=True):
    pattern = os.path.join(folder_path, f"*_{side}_*.png")
    image_files = sorted(glob.glob(pattern))
    if len(image_files) == 0:
        if verbose:
            print(f"No images found for side {side} in {folder_path}")
        return 0, 0, 0

    aggregated_area = 0
    layers_with_area = []
    for img_path in image_files:
        base = os.path.basename(img_path)
        try:
            layer_str = base.split('_')[-1].split('.')[0]
            layer = int(layer_str)
        except Exception as e:
            layer = 0
        
        img_tensor, orig_shape = preprocess_image(img_path, input_size)
        if img_tensor is None:
            continue
        mask = ensemble_segmentation(models, img_tensor, device)
        area = np.sum(mask)
        aggregated_area += area
        if area > 0:
            layers_with_area.append(layer)
        
        if verbose:
            print(f"File: {base} | Orig shape: {orig_shape} | Layer: {layer} | Area: {area}")
        
        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        save_artifact(orig_img, mask, area, layer, os.path.basename(folder_path), side, artifact_dir)
    
    start_layer = min(layers_with_area) if layers_with_area else 0
    end_layer = max(layers_with_area) if layers_with_area else 0
    return aggregated_area, start_layer, end_layer

# -----------------------------
# 7. Process Test Samples Using Test.csv
# -----------------------------
def process_test_samples(test_csv_path, test_data_dir, models, device='cpu', input_size=(640,640), artifact_dir="artifacts", verbose=True):
    test_df = pd.read_csv(test_csv_path)
    submission_records = []
    # Assume Test.csv has columns: ID, FolderName, PlantNumber, Side, Genotype, Stage
    for idx, row in test_df.iterrows():
        sample_id = row["ID"]
        folder = row["FolderName"]
        plant_number = row["PlantNumber"] if "PlantNumber" in row else 1
        side = row["Side"]
        genotype = row["Genotype"] if "Genotype" in row else "Unknown"
        stage = row["Stage"] if "Stage" in row else "Unknown"
        folder_path = os.path.join(test_data_dir, folder)
        
        if not os.path.exists(folder_path):
            if verbose:
                print(f"Folder {folder_path} not found. Using default predictions.")
            volume, start_layer, end_layer = 0, 0, 0
        else:
            volume, start_layer, end_layer = process_folder_side(folder_path, side, models, device, input_size, artifact_dir, verbose)
        
        record = {
            "ID": sample_id,
            "FolderName": folder,
            "PlantNumber": plant_number,
            "Side": side,
            "Start": start_layer,
            "End": end_layer,
            "Genotype": genotype,
            "Stage": stage,
            "RootVolume": volume
        }
        submission_records.append(record)
        print(f"Processed Sample ID: {sample_id} | Folder: {folder} | Side: {side} | Volume: {volume} | Layer range: {start_layer}-{end_layer}")
    
    submission_df = pd.DataFrame(submission_records)
    return submission_df

# -----------------------------
# 8. Main Execution Block
# -----------------------------
if __name__ == "__main__":
    # Set paths in WSL2 style
    test_csv_path = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Test.csv"
    test_data_dir = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/test"
    artifact_dir = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/artifacts"
    model_paths = [
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_early.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_late.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_full.pt"
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models = []
    for mp in model_paths:
        print(f"Loading model from {mp}")
        model = load_model(mp, device)
        models.append(model)
    print("All models loaded. Starting ensemble inference pipeline using Test.csv...")
    
    submission_df = process_test_samples(test_csv_path, test_data_dir, models, device, input_size=(640,640), artifact_dir=artifact_dir, verbose=True)
    submission_csv_path = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_ensemble_2.csv"
    submission_df.to_csv(submission_csv_path, index=False)
    print(f"Submission file saved to {submission_csv_path}")