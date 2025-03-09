import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from math import sqrt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

#####################################
# CONFIGURATION
#####################################
TRAIN_CSV_PATH = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Train.csv"
TRAIN_DATA_DIR = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/train"
TEST_CSV_PATH = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Test.csv"
TEST_DATA_DIR = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/test"
MODEL_SAVE_PATH = "../volume_correction_nn.pth"
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LAYER_THICKNESS = 0.1
INPUT_SIZE = (640, 640)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#####################################
# IMAGE PROCESSING UTILITIES
#####################################
def letterbox_resize(img, new_shape=(640, 640)):
    shape = img.shape[:2]
    ratio = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(shape[1]*ratio), int(shape[0]*ratio))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    top, bottom = dh//2, dh-(dh//2)
    left, right = dw//2, dw-(dw//2)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))

def load_model(model_path):
    """Load YOLO model with proper device handling"""
    model = torch.load(model_path, map_location=torch.device(DEVICE))
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    return model.eval().half().to(DEVICE)

def preprocess_image(img_path, rotate=False):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_180)
    img = letterbox_resize(img)
    img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0) / 255.0
    return img_tensor.half().to(DEVICE), img.shape[:2]

def ensemble_segmentation(models, img_tensor):
    """Combine predictions from multiple YOLO models"""
    masks = []
    for model in models:
        with torch.no_grad():
            output = model(img_tensor)
            mask = output[0] if isinstance(output, tuple) else output
            mask = mask.squeeze().cpu().numpy()
            masks.append((mask > 0.2).astype(np.uint8))
    return np.mean(masks, axis=0) > 0.5

#####################################
# VOLUME CALCULATION CORE
#####################################
def simpson_integration(areas, dx):
    n = len(areas)
    if n < 2: return 0.0
    if n < 3 or n%2 == 0: return np.trapz(areas, dx=dx)
    return (dx/3)*(areas[0] + areas[-1] + 4*np.sum(areas[1:-1:2]) + 2*np.sum(areas[2:-2:2]))

def process_side(folder_path, side, models):
    pattern = os.path.join(folder_path, f"*_[{side.lower()}{side.upper()}]_*.png")
    files = sorted(glob.glob(pattern))
    
    layer_areas = []
    for f in files:
        try:
            layer = int(os.path.basename(f).split('_')[-1].split('.')[0])
            img_tensor, _ = preprocess_image(f, rotate=(side.lower()=='r'))
            if img_tensor is None: continue
            
            mask = ensemble_segmentation(models, img_tensor)
            layer_areas.append((layer, mask.sum()))
        except Exception as e:
            print(f"Error processing {f}: {str(e)}")
    
    if not layer_areas: return 0, 0, []
    layers, areas = zip(*sorted(layer_areas, key=lambda x: x[0]))
    return layers[0], layers[-1], areas

def calculate_volume_features(folder_path, models):
    """Enhanced feature engineering with multiple integration methods"""
    try:
        left_results = process_side(folder_path, 'L', models)
        right_results = process_side(folder_path, 'R', models)
        
        # Base Simpson's integration (main feature)
        simpson_left = simpson_integration(left_results[2], LAYER_THICKNESS)
        simpson_right = simpson_integration(right_results[2], LAYER_THICKNESS)
        raw_volume = (simpson_left + simpson_right) / 2
        
        # Additional features
        features = {
            'raw_volume': raw_volume,
            'depth': (left_results[1] - left_results[0] + right_results[1] - right_results[0])/2,
            'num_layers': (len(left_results[2]) + len(right_results[2]))/2,
            'area_variance': np.var(left_results[2] + right_results[2]),
            'max_area': max(left_results[2] + right_results[2]),
            'min_area': min(left_results[2] + right_results[2])
        }
        return features
    except Exception as e:
        print(f"Error processing {folder_path}: {str(e)}")
        return None

#####################################
# ADVANCED NEURAL NETWORK
#####################################
class VolumeCorrector(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
        self.main_net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        attn_weights = self.attention(x)
        return self.main_net(x * attn_weights)

#####################################
# ROBUST PIPELINE
#####################################
class RootVolumePipeline:
    def __init__(self):
        self.models = []
        self.scaler = StandardScaler()
        self.corrector = None
        self.feature_names = None
        
    def load_models(self, model_paths):
        print("Loading YOLO models...")
        self.models = [load_model(p) for p in tqdm(model_paths)]
        print(f"Loaded {len(self.models)} models")
        
    def prepare_training_data(self):
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        features, labels = [], []
        
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing training data"):
            folder_path = os.path.join(TRAIN_DATA_DIR, row['FolderName'])
            feat = calculate_volume_features(folder_path, self.models)
            if feat is None: continue
            
            if self.feature_names is None:
                self.feature_names = list(feat.keys())
            features.append([feat[k] for k in self.feature_names])
            labels.append(row['RootVolume'])
        
        X = np.array(features)
        y = np.array(labels)
        self.scaler.fit(X)
        return self.scaler.transform(X), y
    
    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.corrector = VolumeCorrector(X.shape[1]).to(DEVICE)
        optimizer = optim.AdamW(self.corrector.parameters(), lr=1e-4, weight_decay=1e-5)
        best_rmse = float('inf')
        
        for epoch in range(EPOCHS):
            self.corrector.train()
            epoch_loss = 0
            for X_batch, y_batch in DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), 
                                       batch_size=BATCH_SIZE, shuffle=True):
                optimizer.zero_grad()
                outputs = self.corrector(X_batch.to(DEVICE)).squeeze()
                loss = nn.HuberLoss()(outputs, y_batch.to(DEVICE))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.corrector.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.corrector.eval()
            with torch.no_grad():
                val_pred = self.corrector(torch.FloatTensor(X_val).to(DEVICE)).cpu().numpy()
                rmse = sqrt(mean_squared_error(y_val, val_pred))
                if rmse < best_rmse:
                    best_rmse = rmse
                    torch.save(self.corrector.state_dict(), MODEL_SAVE_PATH)
                    print(f"Epoch {epoch+1:03d} | New best RMSE: {rmse:.4f}")
    
    def predict(self, folder_path):
        feat = calculate_volume_features(folder_path, self.models)
        if feat is None: return 0.0
        features = np.array([[feat[k] for k in self.feature_names]])
        features = self.scaler.transform(features)
        with torch.no_grad():
            return max(self.corrector(torch.FloatTensor(features).to(DEVICE)).item(), 0)

#####################################
# MAIN EXECUTION
#####################################
if __name__ == "__main__":
    pipeline = RootVolumePipeline()
    model_paths = [
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_early.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_late.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_full.pt"
    ]
    
    # Load YOLO models
    pipeline.load_models(model_paths)
    
    # Train or load correction model
    if os.path.exists(MODEL_SAVE_PATH):
        pipeline.corrector = VolumeCorrector(len(pipeline.feature_names)).to(DEVICE)
        pipeline.corrector.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        X, y = pipeline.prepare_training_data()
        pipeline.train(X, y)
    
    # Generate predictions
    test_df = pd.read_csv(TEST_CSV_PATH)
    predictions = []
    for _, row in tqdm(test_df.iterrows(), desc="Processing test data"):
        folder_path = os.path.join(TEST_DATA_DIR, row['FolderName'])
        predictions.append(pipeline.predict(folder_path))
    
    test_df['RootVolume'] = predictions
    test_df.to_csv("final_submission.csv", index=False)
    print("Submission saved successfully")