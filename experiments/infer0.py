import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd

# -----------------------------
# 1. Load the YOLO v11 model
# -----------------------------
def load_model(model_path, device='cpu'):
    # For a custom model, adjust loading as needed.
    # Here we assume the model can be loaded directly via torch.load
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    return model

# -----------------------------
# 2. Preprocess image (if necessary)
# -----------------------------
def preprocess_image(img_path, input_size=(640, 640)):
    # Load image and resize; adjust preprocessing as required by your model
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, input_size)
    # Convert to tensor and normalize (example; adjust normalization as per model training)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, H, W]
    return img_tensor, img

# -----------------------------
# 3. Run inference and get segmentation output
# -----------------------------
def run_inference(model, img_tensor, device='cpu'):
    with torch.no_grad():
        # The output structure depends on your model implementation.
        # Here we assume it returns a list of detections with bounding boxes and segmentation masks.
        output = model(img_tensor.to(device))
    return output

# -----------------------------
# 4. Process inference results
# -----------------------------
def process_detections(output, threshold=0.5):
    # This function processes the model output.
    # For a baseline, assume each detection contains a confidence and a segmentation mask or bounding box.
    # Here, we will simply return the total area detected (as a proxy for root volume) for each detection.
    # Adjust according to your model's output structure.
    detected_areas = []
    # Dummy processing: iterate through detections and sum areas above a threshold
    # (Replace with actual parsing logic)
    for detection in output:
        conf = detection.get('conf', 0)
        if conf >= threshold:
            # Assuming each detection has a segmentation mask as a numpy array.
            seg_mask = detection.get('mask', None)
            if seg_mask is not None:
                area = np.sum(seg_mask > 0)  # Count nonzero pixels
                detected_areas.append(area)
    return detected_areas

# -----------------------------
# 5. Estimate volume for a side (L or R) in a folder
# -----------------------------
def estimate_volume_for_side(folder_path, side, model, device='cpu'):
    # Get all images for the side (L or R)
    pattern = os.path.join(folder_path, f"*_{side}_*.png")
    image_files = sorted(glob.glob(pattern))
    
    # For baseline: use the image with the maximum total detected area as the representative image.
    best_area = 0
    best_start = None
    best_end = None
    for img_path in image_files:
        # Extract layer number from filename (assumes filename format includes _NNN.png)
        base = os.path.basename(img_path)
        layer_str = base.split('_')[-1].split('.')[0]
        layer = int(layer_str)
        
        img_tensor, _ = preprocess_image(img_path)
        output = run_inference(model, img_tensor, device)
        detected_areas = process_detections(output)
        total_area = sum(detected_areas) if detected_areas else 0
        
        # Update best area and record the layer as both start and end for now.
        if total_area > best_area:
            best_area = total_area
            best_start = layer
            best_end = layer
    
    # For a baseline, we use the best image's detected area as a proxy for volume.
    # In a more refined approach, you might aggregate over a range of layers.
    return best_area, best_start, best_end

# -----------------------------
# 6. Main pipeline for creating submission CSV
# -----------------------------
def create_submission(test_data_dir, model, device='cpu'):
    submission_records = []
    # Assuming each subfolder in test_data_dir is one folder for scanning a set of plants.
    folder_list = sorted([f for f in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, f))])
    
    record_id = 0
    for folder in folder_list:
        folder_path = os.path.join(test_data_dir, folder)
        # Process both sides: Left (L) and Right (R)
        for side in ['L', 'R']:
            # Estimate volume from the best layer/image for this side.
            volume, start_layer, end_layer = estimate_volume_for_side(folder_path, side, model, device)
            
            # For simplicity, assume that for each side the detected plants (numbered left-to-right) are combined.
            # Here we assume one entry per side per folder. For a refined pipeline, you would separate multiple plants.
            record = {
                'ID': record_id,
                'FolderName': folder,
                'PlantNumber': 1,  # Baseline: assume one plant per side; adjust if you detect multiple.
                'Side': side,
                'Start': start_layer if start_layer is not None else 0,
                'End': end_layer if end_layer is not None else 0,
                'Genotype': 'Unknown',  # Update if you have genotype info.
                'Stage': 'Unknown',     # Update if you have stage info.
                'RootVolume': volume
            }
            submission_records.append(record)
            record_id += 1

    submission_df = pd.DataFrame(submission_records)
    return submission_df

# -----------------------------
# 7. Run the pipeline and save submission CSV
# -----------------------------
if __name__ == "__main__":
    # Paths to the model and test data directory.
    model_path = 'Models/best_full.pt'
    test_data_dir = 'data/test'  # Adjust this to your local test images folder
    submission_csv_path = 'submission.csv'
    
    # Load model (using CPU or CUDA if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    
    # Create submission DataFrame
    submission_df = create_submission(test_data_dir, model, device)
    
    # Save to CSV (ensure that the CSV structure matches the sample submission)
    submission_df.to_csv(submission_csv_path, index=False)
    
    print(f"Submission file saved to {submission_csv_path}")
