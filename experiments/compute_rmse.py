import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def compute_rmse(pred_csv, truth_csv):
    pred_df = pd.read_csv(pred_csv)
    truth_df = pd.read_csv(truth_csv)

    # Ensure column names are stripped of spaces
    pred_df.columns = pred_df.columns.str.strip()
    truth_df.columns = truth_df.columns.str.strip()



    pred_df['PlantNumber'] = pd.to_numeric(pred_df['PlantNumber'], errors='coerce')
    truth_df['PlantNumber'] = pd.to_numeric(truth_df['PlantNumber'], errors='coerce')


    print("Extra in Predictions:", set(pred_df['FolderName']) - set(truth_df['FolderName']))
    print("Extra in Ground Truth:", set(truth_df['FolderName']) - set(pred_df['FolderName']))


    pred_df.rename(columns={'RootVolume': 'RootVolume_x'}, inplace=True)
    truth_df.rename(columns={'RootVolume': 'RootVolume_y'}, inplace=True)



    print("Prediction Sample:\n", pred_df[['FolderName', 'PlantNumber', 'Side']].head())
    print("Ground Truth Sample:\n", truth_df[['FolderName', 'PlantNumber', 'Side']].head())

    print("Unique FolderNames in Predictions:", pred_df['FolderName'].unique()[:10])
    print("Unique FolderNames in Ground Truth:", truth_df['FolderName'].unique()[:10])

    print("Unique PlantNumbers in Predictions:", pred_df['PlantNumber'].unique()[:10])
    print("Unique PlantNumbers in Ground Truth:", truth_df['PlantNumber'].unique()[:10])

    missing_in_truth = set(pred_df['FolderName']) - set(truth_df['FolderName'])
    missing_in_pred = set(truth_df['FolderName']) - set(pred_df['FolderName'])

    print("Missing in Ground Truth:", missing_in_truth)
    print("Missing in Predictions:", missing_in_pred)

    # Standardize types and remove extra spaces
    for col in ['FolderName', 'PlantNumber', 'Side']:
        pred_df[col] = pred_df[col].astype(str).str.strip().str.lower()
        truth_df[col] = truth_df[col].astype(str).str.strip().str.lower()


    # Diagnose mismatches
    merged_df = pd.merge(pred_df, truth_df, on=['FolderName', 'PlantNumber', 'Side'], how='outer', indicator=True)
    print(merged_df['_merge'].value_counts())

    # Keep only valid matches
    merged_df = merged_df[merged_df['_merge'] == 'both']

    if merged_df.empty:
        print("No matching entries found after cleaning.")
        return None

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(merged_df['RootVolume_x'], merged_df['RootVolume_y']))
    print("RMSE:", rmse)
    
    return rmse

if __name__ == "__main__":
    pred_csv = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_regression.csv"
    truth_csv = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Train.csv"




    compute_rmse(pred_csv, truth_csv)
