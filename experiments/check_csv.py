import pandas as pd

pred_csv = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/submission_regression.csv"
truth_csv = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Train.csv"

pred_df = pd.read_csv(pred_csv)
truth_df = pd.read_csv(truth_csv)

print("Prediction CSV Columns:", pred_df.columns)
print("Ground Truth CSV Columns:", truth_df.columns)
