import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

#############################
# SEGMENTATION PIPELINE CODE
#############################

def load_model(model_path, device='cpu'):
    model = torch.load(model_path, map_location=torch.device(device))
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    model.eval()
    return model

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

def get_segmentation_mask(model, img_tensor, device='cpu'):
    with torch.no_grad():
        # Convert input to half precision to match FP16 model weights
        output = model(img_tensor.to(device).half())
        # Provide more detailed logging about output type
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
        # Assume mask shape is [1, 1, H, W]; squeeze to [H, W]
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)
    return binary_mask

def ensemble_segmentation(models, img_tensor, device='cpu'):
    masks = []
    for model in models:
        mask = get_segmentation_mask(model, img_tensor, device)
        masks.append(mask)
    avg_mask = np.mean(masks, axis=0)
    binary_avg_mask = (avg_mask > 0.5).astype(np.uint8)
    return binary_avg_mask

# For faster processing, we do not display artifacts here.
def process_folder_side(folder_path, side, models, device='cpu', input_size=(640, 640), verbose=True):
    pattern = os.path.join(folder_path, f"*_{side}_*.png")
    image_files = sorted(glob.glob(pattern))
    if len(image_files) == 0:
        if verbose:
            print(f"No images found for side {side} in {folder_path}")
        return 0, 0, 0  # default features

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
    
    start_layer = min(layers_with_area) if layers_with_area else 0
    end_layer = max(layers_with_area) if layers_with_area else 0
    return aggregated_area, start_layer, end_layer

#########################################
# FEATURE EXTRACTION AND REGRESSION CODE
#########################################

def extract_features_from_sample(folder, side, models, test_data_dir, device='cpu', input_size=(640,640), verbose=False):
    """
    For a given sample (identified by FolderName and Side), run the segmentation pipeline
    and extract features. Features include:
      - aggregated_area: Sum of white pixels (proxy for root area)
      - start_layer: Minimum layer with detected area
      - end_layer: Maximum layer with detected area
      - num_layers: (end_layer - start_layer + 1) if nonzero
      - avg_area: aggregated_area / num_layers (if num_layers > 0)
    """
    folder_path = os.path.join(test_data_dir, folder)
    if not os.path.exists(folder_path):
        if verbose:
            print(f"Folder {folder_path} not found. Using default features.")
        return {
            'aggregated_area': 0,
            'start_layer': 0,
            'end_layer': 0,
            'num_layers': 0,
            'avg_area': 0
        }
    aggregated_area, start_layer, end_layer = process_folder_side(folder_path, side, models, device, input_size, verbose)
    num_layers = (end_layer - start_layer + 1) if (start_layer and end_layer) else 0
    avg_area = aggregated_area / num_layers if num_layers > 0 else 0
    features = {
        'aggregated_area': aggregated_area,
        'start_layer': start_layer,
        'end_layer': end_layer,
        'num_layers': num_layers,
        'avg_area': avg_area
    }
    if verbose:
        print(f"Extracted features for Folder: {folder}, Side: {side} -> {features}")
    return features

def build_training_features(train_csv_path, models, train_data_dir, device='cpu', input_size=(640,640), verbose=False):
    # Read Train.csv; assumed to have columns: ID, FolderName, PlantNumber, Side, Genotype, Stage, RootVolume
    train_df = pd.read_csv(train_csv_path)
    features_list = []
    # Use a cache so that if the same folder/side appears multiple times, we only process once
    feature_cache = {}
    for idx, row in train_df.iterrows():
        folder = row["FolderName"]
        side = row["Side"]
        key = (folder, side)
        if key not in feature_cache:
            feat = extract_features_from_sample(folder, side, models, train_data_dir, device, input_size, verbose)
            feature_cache[key] = feat
        else:
            feat = feature_cache[key]
        # Build feature vector and include target (RootVolume)
        features_list.append({
            'aggregated_area': feat['aggregated_area'],
            'start_layer': feat['start_layer'],
            'end_layer': feat['end_layer'],
            'num_layers': feat['num_layers'],
            'avg_area': feat['avg_area'],
            'RootVolume': row['RootVolume']
        })
    feature_df = pd.DataFrame(features_list)
    if verbose:
        print("Training feature DataFrame head:")
        print(feature_df.head())
    return feature_df

def build_test_features(test_csv_path, models, test_data_dir, device='cpu', input_size=(640,640), verbose=False):
    # Read Test.csv; assumed to have columns: ID, FolderName, PlantNumber, Side, Genotype, Stage
    test_df = pd.read_csv(test_csv_path)
    features_list = []
    feature_cache = {}
    for idx, row in test_df.iterrows():
        folder = row["FolderName"]
        side = row["Side"]
        key = (folder, side)
        if key not in feature_cache:
            feat = extract_features_from_sample(folder, side, models, test_data_dir, device, input_size, verbose)
            feature_cache[key] = feat
        else:
            feat = feature_cache[key]
        features_list.append({
            'ID': row['ID'],
            'FolderName': folder,
            'PlantNumber': row.get('PlantNumber', 1),
            'Side': side,
            'aggregated_area': feat['aggregated_area'],
            'start_layer': feat['start_layer'],
            'end_layer': feat['end_layer'],
            'num_layers': feat['num_layers'],
            'avg_area': feat['avg_area'],
            'Genotype': row.get('Genotype', 'Unknown'),
            'Stage': row.get('Stage', 'Unknown')
        })
    feature_df = pd.DataFrame(features_list)
    if verbose:
        print("Test feature DataFrame head:")
        print(feature_df.head())
    return feature_df

#############################
# MAIN TRAINING AND PREDICTION
#############################
if __name__ == "__main__":
    # Set paths in WSL2 style
    # Adjust these paths as needed:
    train_csv_path = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Train.csv"
    test_csv_path  = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Test.csv"
    train_data_dir = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/train"
    test_data_dir  = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/test"
    
    # Model paths (update if needed)
    model_paths = [
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_early.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_late.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_full.pt"
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = (640, 640)
    verbose = True

    # Load all three models for ensembling
    models = []
    for mp in model_paths:
        print(f"Loading model from {mp}")
        model = load_model(mp, device)
        models.append(model)
    print("All models loaded. Extracting training features...")

    # Build training features DataFrame using Train.csv
    train_features = build_training_features(train_csv_path, models, train_data_dir, device, input_size, verbose)
    
    # Define the feature columns and target
    feature_cols = ['aggregated_area', 'start_layer', 'end_layer', 'num_layers', 'avg_area']
    target_col = 'RootVolume'
    
    # Train a regression model (here we use RandomForestRegressor)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train_features[feature_cols], train_features[target_col])
    train_preds = rf.predict(train_features[feature_cols])
    rmse_train = sqrt(mean_squared_error(train_features[target_col], train_preds))
    print(f"Training RMSE: {rmse_train}")
    
    # Build test features DataFrame using Test.csv
    print("Extracting test features...")
    test_features = build_test_features(test_csv_path, models, test_data_dir, device, input_size, verbose)
    
    # Predict using the regression model
    test_preds = rf.predict(test_features[feature_cols])
    test_features['RootVolume'] = test_preds
    
    # Prepare final submission DataFrame; include necessary columns
    submission_df = test_features[['ID', 'FolderName', 'PlantNumber', 'Side', 'start_layer', 'end_layer', 'Genotype', 'Stage', 'RootVolume']]
    submission_df.rename(columns={'start_layer': 'Start', 'end_layer': 'End'}, inplace=True)
    submission_csv_path = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_regression.csv"
    submission_df.to_csv(submission_csv_path, index=False)
    print(f"Submission file saved to {submission_csv_path}")