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
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, H, W]
    return img_tensor, original_shape

def get_segmentation_mask(model, img_tensor, device='cpu'):
    with torch.no_grad():
        # Convert input to half precision (if needed)
        output = model(img_tensor.to(device).half())
        if isinstance(output, tuple):
            # Log output type for debugging
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
        # Assume mask shape [1, 1, H, W] and squeeze to [H, W]
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

def process_folder_side(folder_path, side, models, device='cpu', input_size=(640, 640), verbose=True):
    pattern = os.path.join(folder_path, f"*_{side}_*.png")
    image_files = sorted(glob.glob(pattern))
    if len(image_files) == 0:
        if verbose:
            print(f"No images found for side {side} in {folder_path}. Using default features.")
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

def extract_features_from_sample(folder, side, models, data_dir, device='cpu', input_size=(640,640), verbose=False):
    """
    For a given sample (identified by FolderName and Side), run the segmentation pipeline
    and extract features.
    Returns a dictionary with:
      - aggregated_area: Sum of white pixels (proxy for root area)
      - start_layer: Minimum layer with detected area
      - end_layer: Maximum layer with detected area
      - num_layers: (end_layer - start_layer + 1) if both nonzero, else 0
      - avg_area: aggregated_area / num_layers (if num_layers > 0) else 0
    """
    # Standardize folder and side keys (lowercase, stripped)
    folder_std = folder.strip().lower()
    side_std = side.strip().lower()
    folder_path = os.path.join(data_dir, folder_std)
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
    aggregated_area, start_layer, end_layer = process_folder_side(folder_path, side_std, models, device, input_size, verbose)
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
        print(f"Extracted features for Folder: {folder_std}, Side: {side_std} -> {features}")
    return features

def build_training_features(train_csv_path, models, data_dir, device='cpu', input_size=(640,640), verbose=False):
    """
    Reads Train.csv and extracts segmentation-based features for each sample from the training images.
    Assumes Train.csv has columns: ID, FolderName, PlantNumber, Side, Genotype, Stage, RootVolume.
    Returns a DataFrame with one row per row in Train.csv.
    """
    train_df = pd.read_csv(train_csv_path)
    # Standardize keys in train_df for consistency
    train_df['FolderName'] = train_df['FolderName'].astype(str).str.strip().str.lower()
    train_df['Side'] = train_df['Side'].astype(str).str.strip().str.lower()
    # Ensure PlantNumber is numeric
    train_df['PlantNumber'] = pd.to_numeric(train_df['PlantNumber'], errors='coerce')
    
    features_list = []
    # Use a cache so that if the same (FolderName, Side) appears multiple times, we compute features only once
    feature_cache = {}
    for idx, row in train_df.iterrows():
        folder = row["FolderName"]
        side = row["Side"]
        key = (folder, side)
        if key not in feature_cache:
            feat = extract_features_from_sample(folder, side, models, data_dir, device, input_size, verbose)
            feature_cache[key] = feat
        else:
            feat = feature_cache[key]
        # Build feature dictionary (include the target for training)
        feat_dict = {
            'aggregated_area': feat['aggregated_area'],
            'start_layer': feat['start_layer'],
            'end_layer': feat['end_layer'],
            'num_layers': feat['num_layers'],
            'avg_area': feat['avg_area'],
            'RootVolume': row['RootVolume']
        }
        features_list.append(feat_dict)
    feature_df = pd.DataFrame(features_list)
    if verbose:
        print("Training feature DataFrame head:")
        print(feature_df.head())
    return feature_df, train_df  # return train_df to preserve order and metadata

#########################################
# MAIN TRAINING AND PREDICTION ON TRAIN SET
#########################################

if __name__ == "__main__":
    # Set paths for the training set
    train_csv_path = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Train.csv"
    data_dir = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/train"
    
    # Model paths
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
    print("All models loaded. Extracting training features from Train.csv...")

    # Build training features (feature_df will have one row per row in Train.csv)
    feature_df, train_df_std = build_training_features(train_csv_path, models, data_dir, device, input_size, verbose)
    
    # Define feature columns and target
    feature_cols = ['aggregated_area', 'start_layer', 'end_layer', 'num_layers', 'avg_area']
    target_col = 'RootVolume'
    
    # Train a regression model (RandomForestRegressor)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(feature_df[feature_cols], feature_df[target_col])
    train_preds = rf.predict(feature_df[feature_cols])
    rmse_train = sqrt(mean_squared_error(feature_df[target_col], train_preds))
    print(f"Training RMSE: {rmse_train}")
    
    # Add the predictions to the feature DataFrame
    feature_df['PredictedRootVolume'] = train_preds

    # Create a submission DataFrame that preserves all rows from Train.csv using its order and metadata.
    # We use train_df_std (the standardized Train.csv) for metadata.
    submission_df = pd.DataFrame({
        'ID': train_df_std['ID'],
        'FolderName': train_df_std['FolderName'],
        'PlantNumber': train_df_std['PlantNumber'],
        'Side': train_df_std['Side'],
        'Start': feature_df['start_layer'],
        'End': feature_df['end_layer'],
        'Genotype': train_df_std['Genotype'],
        'Stage': train_df_std['Stage'],
        'RootVolume': feature_df['PredictedRootVolume']
    })
    
    # Save the submission file (ensuring all Train.csv rows are included)
    submission_csv_path = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_regression_train_2.csv"
    submission_df.to_csv(submission_csv_path, index=False)
    print(f"Training submission file saved to {submission_csv_path}")