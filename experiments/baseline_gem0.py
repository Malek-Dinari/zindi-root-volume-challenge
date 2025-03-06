import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd

# -----------------------------
# 1. Model Loading
# -----------------------------
def load_model(model_path, device='cpu'):
    # Load the model (set weights_only=True if applicable for security)
    model = torch.load(model_path, map_location=torch.device(device))
    # In many cases, the loaded file might be a dict (e.g., checkpoint); adjust accordingly.
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    model.eval()
    return model

# -----------------------------
# 2. Preprocess Image
# -----------------------------
def preprocess_image(img_path, input_size=(640, 640)):
    # Read image with OpenCV (BGR)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_path}")
        return None, None
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]  # (height, width)
    # Resize image for model input
    img_resized = cv2.resize(img, input_size)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]
    return img_tensor, original_shape

# -----------------------------
# 3. Run Inference on a Single Model
# -----------------------------
def get_segmentation_mask(model, img_tensor, device='cpu'):
    with torch.no_grad():
        output = model(img_tensor.to(device).half())
        print(type(output), len(output) if isinstance(output, tuple) else output.shape)
        output = output[0]
        # Example: if the model returns a dictionary with key 'mask'
        if isinstance(output, dict):
            mask = output.get('mask', None)
            if mask is None:
                mask = output.get('pred', None)
        else:
            mask = output
        # For this example, we assume output mask shape is [1, 1, H, W] and contains probabilities.
        if isinstance(mask, tuple):
            mask = mask[0]  # Extract the first tensor if it's a tuple
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
        # Threshold to get binary mask; adjust threshold if needed.
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
    # Average the masks pixel-wise and then threshold
    avg_mask = np.mean(masks, axis=0)
    binary_avg_mask = (avg_mask > 0.5).astype(np.uint8)
    return binary_avg_mask

# -----------------------------
# 5. Process a Folder's Side (L or R)
# -----------------------------
def process_folder_side(folder_path, side, models, device='cpu', input_size=(640, 640)):
    # List images corresponding to the given side (L or R)
    pattern = os.path.join(folder_path, f"*_{side}_*.png")
    image_files = sorted(glob.glob(pattern))
    if len(image_files) == 0:
        return 0, None, None

    aggregated_area = 0
    layers_with_area = []
    for img_path in image_files:
        base = os.path.basename(img_path)
        try:
            # Extract layer number from filename (assumes format: XXXXXXXX_S_NNN.png)
            layer_str = base.split('_')[-1].split('.')[0]
            layer = int(layer_str)
        except Exception as e:
            layer = None

        img_tensor, orig_shape = preprocess_image(img_path, input_size)
        if img_tensor is None:
            continue
        # Run ensemble inference to get binary segmentation mask
        mask = ensemble_segmentation(models, img_tensor, device)
        # Calculate the area (number of white pixels in the mask)
        area = np.sum(mask)
        aggregated_area += area
        if area > 0 and layer is not None:
            layers_with_area.append(layer)

    # If any layers had detected roots, record the range (start and end layers)
    if layers_with_area:
        start_layer = min(layers_with_area)
        end_layer = max(layers_with_area)
    else:
        start_layer = None
        end_layer = None
    # Here, the aggregated area serves as a proxy for root volume
    return aggregated_area, start_layer, end_layer

# -----------------------------
# 6. Process All Folders & Prepare Submission DataFrame
# -----------------------------
def process_all_folders(test_data_dir, models, device='cpu', input_size=(640, 640)):
    submission_records = []
    folder_list = sorted([f for f in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, f))])
    record_id = 0
    for folder in folder_list:
        folder_path = os.path.join(test_data_dir, folder)
        for side in ['L', 'R']:
            volume, start_layer, end_layer = process_folder_side(folder_path, side, models, device, input_size)
            record = {
                'ID': record_id,
                'FolderName': folder,
                'PlantNumber': 1,  # For baseline, assume one plant per side; refine if needed.
                'Side': side,
                'Start': start_layer if start_layer is not None else 0,
                'End': end_layer if end_layer is not None else 0,
                'Genotype': 'Unknown',  # Update if genotype information is available.
                'Stage': 'Unknown',     # Update if stage information is available.
                'RootVolume': volume
            }
            submission_records.append(record)
            record_id += 1
    submission_df = pd.DataFrame(submission_records)
    return submission_df

# -----------------------------
# 7. Main Execution Block
# -----------------------------
if __name__ == "__main__":
    # WSL2 style paths (adjust these to match your local directories)
    test_data_dir = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/test"
    model_paths = [
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_early.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_late.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_full.pt"
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load all three models for ensembling
    models = []
    for mp in model_paths:
        print(f"Loading model from {mp}")
        model = load_model(mp, device)
        models.append(model)
    print("All models loaded. Starting the ensemble inference pipeline...")

    # Process all folders and build the submission DataFrame
    submission_df = process_all_folders(test_data_dir, models, device, input_size=(640, 640))
    submission_csv_path = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_ensemble.csv"
    submission_df.to_csv(submission_csv_path, index=False)
    print(f"Submission file saved to {submission_csv_path}")