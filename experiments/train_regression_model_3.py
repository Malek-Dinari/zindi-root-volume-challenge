import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from joblib import dump, load

#####################################
# SEGMENTATION PIPELINE FUNCTIONS
#####################################

def load_model(model_path, device='cpu'):
    """Loads a YOLO v11 model from file."""
    model = torch.load(model_path, map_location=torch.device(device))
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    model.eval()
    return model

def preprocess_image(img_path, input_size=(640, 640), rotate=False):
    """Reads and preprocesses an image. Rotates it 180° if rotate=True."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_path}")
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_180)
        print(f"Rotated image {os.path.basename(img_path)} by 180 degrees")
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, input_size)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor, original_shape

def get_segmentation_mask(model, img_tensor, device='cpu'):
    """Runs inference on a single image tensor and returns a binary segmentation mask."""
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
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)
    return binary_mask

def ensemble_segmentation(models, img_tensor, device='cpu'):
    """Runs inference using an ensemble of models and returns a binary mask."""
    masks = []
    for model in models:
        mask = get_segmentation_mask(model, img_tensor, device)
        masks.append(mask)
    avg_mask = np.mean(masks, axis=0)
    binary_avg_mask = (avg_mask > 0.5).astype(np.uint8)
    return binary_avg_mask

def process_folder_side(folder_path, side, models, device='cpu', input_size=(640, 640), verbose=True):
    """
    Processes all images in one scan side of a folder.
    If side=='r', images are rotated 180° before segmentation.
    Returns aggregated area, start layer, and end layer.
    """
    rotate_flag = True if side.lower() == 'r' else False
    pattern = os.path.join(folder_path, f"*_{side}_*.png")
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
        except Exception as e:
            layer = 0
        img_tensor, orig_shape = preprocess_image(img_path, input_size, rotate=rotate_flag)
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

def extract_features_from_sample(folder, side, models, data_dir, device='cpu', input_size=(640,640), verbose=False):
    """
    Extracts segmentation features for a sample (identified by FolderName and Side).
    Returns aggregated_area, start_layer, end_layer, num_layers, and avg_area.
    """
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

#########################################
# DATAFRAME BUILDING FUNCTIONS
#########################################

def build_training_features(train_csv_path, models, data_dir, device='cpu', input_size=(640,640), verbose=False):
    """
    Reads Train.csv and extracts features for each training sample.
    Also, for right-side samples, transforms PlantNumber using:
      Transformed = 7 - Original + 1.
    Returns (features DataFrame, standardized Train DataFrame).
    """
    train_df = pd.read_csv(train_csv_path)
    # Standardize keys
    train_df['FolderName'] = train_df['FolderName'].astype(str).str.strip().str.lower()
    train_df['Side'] = train_df['Side'].astype(str).str.strip().str.lower()
    train_df['PlantNumber'] = pd.to_numeric(train_df['PlantNumber'], errors='coerce')
    # Transform right side plant numbers to align with left side
    right_mask = train_df['Side'] == 'r'
    if right_mask.any():
        original_numbers = train_df.loc[right_mask, 'PlantNumber']
        transformed = 7 - original_numbers + 1
        train_df.loc[right_mask, 'PlantNumber'] = transformed
        print("Transformed PlantNumber for right side samples using: 7 - PlantNumber + 1")
    
    features_list = []
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
    return feature_df, train_df

def build_test_features(test_csv_path, models, data_dir, device='cpu', input_size=(640,640), verbose=False):
    """
    Reads Test.csv and extracts features for each test sample.
    Applies the same right-side transformation as in training.
    Returns a feature DataFrame.
    """
    test_df = pd.read_csv(test_csv_path)
    test_df['FolderName'] = test_df['FolderName'].astype(str).str.strip().str.lower()
    test_df['Side'] = test_df['Side'].astype(str).str.strip().str.lower()
    test_df['PlantNumber'] = pd.to_numeric(test_df['PlantNumber'], errors='coerce')
    right_mask = test_df['Side'] == 'r'
    if right_mask.any():
        original_numbers = test_df.loc[right_mask, 'PlantNumber']
        transformed = 7 - original_numbers + 1
        test_df.loc[right_mask, 'PlantNumber'] = transformed
        print("Transformed PlantNumber for right side samples in Test.csv using: 7 - PlantNumber + 1")
    
    features_list = []
    feature_cache = {}
    for idx, row in test_df.iterrows():
        folder = row["FolderName"]
        side = row["Side"]
        key = (folder, side)
        if key not in feature_cache:
            feat = extract_features_from_sample(folder, side, models, data_dir, device, input_size, verbose)
            feature_cache[key] = feat
        else:
            feat = feature_cache[key]
        features_list.append({
            'ID': row['ID'],
            'FolderName': folder,
            'PlantNumber': row['PlantNumber'],
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

#########################################
# MAIN TRAINING AND PREDICTION PIPELINE
#########################################

if __name__ == "__main__":
    # Paths for training and test sets
    train_csv_path = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Train.csv"
    test_csv_path  = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Test.csv"
    train_data_dir = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/train"
    test_data_dir  = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/test"
    
    # Model paths for YOLO models
    model_paths = [
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_early.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_late.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_full.pt"
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = (640, 640)
    verbose = True

    # Load YOLO models for ensemble segmentation
    models = []
    for mp in model_paths:
        print(f"Loading model from {mp}")
        model = load_model(mp, device)
        models.append(model)
    print("All models loaded.")

    #########################################
    # TRAINING PHASE: Feature Extraction & Model Training
    #########################################
    print("Extracting training features from Train.csv...")
    train_feature_df, train_df_std = build_training_features(train_csv_path, models, train_data_dir, device, input_size, verbose)
    feature_cols = ['aggregated_area', 'start_layer', 'end_layer', 'num_layers', 'avg_area']
    target_col = 'RootVolume'
    
    # Use GridSearchCV to tune an XGBoost model
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror')
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.9, 1.0]
    }
    grid = GridSearchCV(xgb, param_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
    grid.fit(train_feature_df[feature_cols], train_feature_df[target_col])
    
    print("Best parameters from GridSearchCV:", grid.best_params_)
    best_model = grid.best_estimator_
    best_cv_rmse = -grid.best_score_
    print(f"Best CV RMSE: {best_cv_rmse}")
    
    # Predict on training set and compute RMSE
    train_preds = best_model.predict(train_feature_df[feature_cols])
    rmse_train = sqrt(mean_squared_error(train_feature_df[target_col], train_preds))
    print(f"Training RMSE: {rmse_train}")
    
    # Save the trained XGBoost model
    dump(best_model, 'xgb_model.joblib')
    print("Trained XGBoost model saved as xgb_model.joblib")
    
    # Build a submission DataFrame for training set (for debugging/evaluation)
    train_feature_df['PredictedRootVolume'] = train_preds
    submission_train_df = pd.DataFrame({
        'ID': train_df_std['ID'],
        'FolderName': train_df_std['FolderName'],
        'PlantNumber': train_df_std['PlantNumber'],
        'Side': train_df_std['Side'],
        'Start': train_feature_df['start_layer'],
        'End': train_feature_df['end_layer'],
        'Genotype': train_df_std['Genotype'],
        'Stage': train_df_std['Stage'],
        'RootVolume': train_feature_df['PredictedRootVolume']
    })
    submission_train_csv = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_regression_train.csv"
    submission_train_df.to_csv(submission_train_csv, index=False)
    print(f"Training submission file saved to {submission_train_csv}")
    
    #########################################
    # TEST PHASE: Feature Extraction & Prediction
    #########################################
    print("Extracting test features from Test.csv...")
    test_feature_df = build_test_features(test_csv_path, models, test_data_dir, device, input_size, verbose)
    test_preds = best_model.predict(test_feature_df[feature_cols])
    test_feature_df['PredictedRootVolume'] = test_preds
    submission_test_df = pd.DataFrame({
        'ID': test_feature_df['ID'],
        'FolderName': test_feature_df['FolderName'],
        'PlantNumber': test_feature_df['PlantNumber'],
        'Side': test_feature_df['Side'],
        'Start': test_feature_df['start_layer'],
        'End': test_feature_df['end_layer'],
        'Genotype': test_feature_df['Genotype'],
        'Stage': test_feature_df['Stage'],
        'RootVolume': test_feature_df['PredictedRootVolume']
    })
    submission_test_csv = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_regression_test.csv"
    submission_test_df.to_csv(submission_test_csv, index=False)
    print(f"Test submission file saved to {submission_test_csv}")