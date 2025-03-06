import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def compute_rmse(pred_csv, truth_csv):
    # Read CSV files
    pred_df = pd.read_csv(pred_csv)
    truth_df = pd.read_csv(truth_csv)

    # Standardize column names (remove extra spaces)
    pred_df.columns = pred_df.columns.str.strip()
    truth_df.columns = truth_df.columns.str.strip()

    # Standardize merge keys: FolderName and Side as lower-case strings
    for col in ['FolderName', 'Side']:
        pred_df[col] = pred_df[col].astype(str).str.strip().str.lower()
        truth_df[col] = truth_df[col].astype(str).str.strip().str.lower()

    # Convert PlantNumber to numeric (int) so that numbers match exactly
    pred_df['PlantNumber'] = pd.to_numeric(pred_df['PlantNumber'], errors='coerce')
    truth_df['PlantNumber'] = pd.to_numeric(truth_df['PlantNumber'], errors='coerce')

    # Rename the RootVolume column to avoid collisions
    pred_df.rename(columns={'RootVolume': 'RootVolume_pred'}, inplace=True)
    truth_df.rename(columns={'RootVolume': 'RootVolume_true'}, inplace=True)

    # Merge on the common keys
    merged_df = pd.merge(pred_df, truth_df, on=['FolderName', 'PlantNumber', 'Side'], how='inner')
    print("Number of merged entries:", len(merged_df))
    
    if merged_df.empty:
        print("No matching entries found. Make sure your prediction file corresponds to the training set!")
        return None

    # Compute RMSE between the ground truth and predictions
    rmse = np.sqrt(mean_squared_error(merged_df['RootVolume_true'], merged_df['RootVolume_pred']))
    print("RMSE:", rmse)
    return rmse

if __name__ == "__main__":
    # IMPORTANT:
    # Use a prediction file generated from processing the training data!
    # For example, if you have saved predictions for training samples as submission_regression_train.csv,
    # then set pred_csv accordingly.
    pred_csv = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_regression_test.csv"
    truth_csv = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Train.csv"
    compute_rmse(pred_csv, truth_csv)