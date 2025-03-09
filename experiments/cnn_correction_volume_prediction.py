import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
from math import sqrt
from tqdm import tqdm
import torchvision.models as models
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

# Configuration
TRAIN_CSV_PATH = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Train.csv"
TRAIN_DATA_DIR = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/train"
TEST_CSV_PATH = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Test.csv"
TEST_DATA_DIR = "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/data/test"
MODEL_SAVE_PATH = "../volume_correction_cnn.pth"
BATCH_SIZE = 16
EPOCHS = 50  # Reduced due to fine-tuning
PATIENCE = 7
LAYER_THICKNESS = 0.1
IMG_SIZE = (128, 128)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image transformations
transform = Compose([
    Resize(IMG_SIZE),
    ToTensor(),
])

def letterbox_resize(img):
    h, w = img.shape[:2]
    new_w = 640
    new_h = 640
    scale = min(new_w/w, new_h/h)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))
    dh, dw = new_h - resized.shape[0], new_w - resized.shape[1]
    return cv2.copyMakeBorder(resized, dh//2, dh-dh//2, dw//2, dw-dw//2, 
                            cv2.BORDER_CONSTANT, value=(114,114,114))

def load_model(model_path):
    model = torch.load(model_path, map_location=DEVICE)
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    return model.eval().half().to(DEVICE)

def ensemble_segmentation(models, img_tensor):
    masks = []
    for model in models:
        with torch.no_grad():
            output = model(img_tensor)
            mask = output[0] if isinstance(output, tuple) else output
            masks.append((mask.squeeze().cpu().numpy() > 0.2).astype(np.uint8))
    return np.mean(masks, axis=0) > 0.5

class RootVolumeDataset(Dataset):
    def __init__(self, df, data_dir, models, scaler=None, train=True):
        self.df = df
        self.data_dir = data_dir
        self.models = models  # Store YOLO models
        self.scaler = scaler or StandardScaler()
        self.train = train
        self.summary_images = []
        self.numerical_features = []
        self.labels = []
        self.prepare_data()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.train:
            return (self.summary_images[idx], 
                    self.numerical_features[idx], 
                    self.labels[idx])
        return (self.summary_images[idx], 
                self.numerical_features[idx])

    def create_summary_image(self, folder_path):
        summary = np.zeros(IMG_SIZE[::-1], dtype=np.float32)
        for side in ['L', 'R']:
            for f in glob.glob(os.path.join(folder_path, f"*_[{side}]_*.png")):
                img_tensor, _ = self.preprocess_image(f, rotate=(side == 'R'))
                if img_tensor is None: continue
                mask = ensemble_segmentation(self.models, img_tensor)
                resized_mask = cv2.resize(mask.astype(np.float32), IMG_SIZE)
                summary += resized_mask
        summary = (summary - summary.min()) / (summary.max() - summary.min() + 1e-8)
        return transform(summary)

    def preprocess_image(self, img_path, rotate=False):
        img = cv2.imread(img_path)
        if img is None: return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rotate: img = cv2.rotate(img, cv2.ROTATE_180)
        img = letterbox_resize(img)
        img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0) / 255.0
        return img_tensor.half().to(DEVICE), img.shape[:2]

    def calculate_numerical_features(self, folder_path):
        def process_side(side):
            files = sorted(glob.glob(os.path.join(folder_path, f"*_[{side}]_*.png")))
            layer_areas = []
            for f in files:
                try:
                    layer = int(f.split('_')[-1].split('.')[0])
                    img_tensor, _ = self.preprocess_image(f, rotate=(side == 'R'))
                    if img_tensor is None: continue
                    mask = ensemble_segmentation(self.models, img_tensor)
                    layer_areas.append((layer, mask.sum()))
                except: continue
            return (np.array([x[0] for x in layer_areas]),
                    np.array([x[1] for x in layer_areas])) if layer_areas else (0, 0, [])
        
        l_start, l_end, l_areas = process_side('L')
        r_start, r_end, r_areas = process_side('R')
        vol = (simpson_integration(l_areas) + simpson_integration(r_areas)) / 2
        return vol, min(l_start, r_start), max(l_end, r_end)

    def prepare_data(self):
        for _, row in tqdm(self.df.iterrows(), desc="Processing data"):
            folder_path = os.path.join(self.data_dir, row['FolderName'])
            summary_img = self.create_summary_image(folder_path)
            self.summary_images.append(summary_img)
            
            vol, start, end = self.calculate_numerical_features(folder_path)
            features = [vol, end-start, start, end]
            self.numerical_features.append(features)
            
            if self.train:
                self.labels.append(row['RootVolume'])
        
        self.numerical_features = self.scaler.fit_transform(self.numerical_features)

class FineTunedCorrector(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        for param in self.cnn.parameters():  # Freeze backbone
            param.requires_grad = False
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.num_fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(256+128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, img, features):
        img_feat = self.cnn(img)
        num_feat = self.num_fc(features)
        return self.head(torch.cat([img_feat, num_feat], dim=1))

class CNNPipeline:
    def __init__(self, model_paths):
        self.models = [load_model(p) for p in model_paths]
        self.scaler = StandardScaler()
        self.corrector = None

    def train(self):
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        train_df, val_df = train_test_split(train_df, test_size=0.2)
        
        # Initialize datasets
        train_set = RootVolumeDataset(train_df, TRAIN_DATA_DIR, self.models, self.scaler)
        val_set = RootVolumeDataset(val_df, TRAIN_DATA_DIR, self.models, self.scaler)
        
        # Initialize model
        self.corrector = FineTunedCorrector(num_features=4).to(DEVICE)
        optimizer = optim.AdamW([
            {'params': self.corrector.cnn.fc.parameters()},
            {'params': self.corrector.num_fc.parameters()},
            {'params': self.corrector.head.parameters()}
        ], lr=1e-4)
        criterion = nn.HuberLoss()
        
        best_rmse = float('inf')
        for epoch in range(EPOCHS):
            self.corrector.train()
            epoch_loss = 0
            for img, feats, label in DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True):
                optimizer.zero_grad()
                pred = self.corrector(img.to(DEVICE), feats.float().to(DEVICE)).squeeze()
                loss = criterion(pred, label.float().to(DEVICE))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.corrector.eval()
            val_pred, val_true = [], []
            with torch.no_grad():
                for img, feats, label in DataLoader(val_set, batch_size=BATCH_SIZE):
                    pred = self.corrector(img.to(DEVICE), feats.float().to(DEVICE)).squeeze().cpu()
                    val_pred.extend(pred.numpy())
                    val_true.extend(label.numpy())
            rmse = sqrt(mean_squared_error(val_true, val_pred))
            
            # Early stopping and checkpointing
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save(self.corrector.state_dict(), MODEL_SAVE_PATH)
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.2f}, Val RMSE={rmse:.2f}")

    def predict(self, test_df):
        test_set = RootVolumeDataset(test_df, TEST_DATA_DIR, self.models, self.scaler, train=False)
        self.corrector.eval()
        preds = []
        with torch.no_grad():
            for img, feats in DataLoader(test_set, batch_size=BATCH_SIZE):
                pred = self.corrector(img.to(DEVICE), feats.float().to(DEVICE)).squeeze().cpu().numpy()
                preds.extend(np.maximum(pred, 0))
        return preds

if __name__ == "__main__":
    model_paths = [
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_early.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_late.pt",
        "/mnt/d/Users/HP/Downloads/datasets/ZINDI-CGIAR-Root-Volume-Estimation-Challenge/Models/best_full.pt"
    ]
    
    pipeline = CNNPipeline(model_paths)
    
    if os.path.exists(MODEL_SAVE_PATH):
        pipeline.corrector = FineTunedCorrector(num_features=4).to(DEVICE)
        pipeline.corrector.load_state_dict(torch.load(MODEL_SAVE_PATH))
    else:
        pipeline.train()
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df['RootVolume'] = pipeline.predict(test_df)
    test_df.to_csv("final_submission.csv", index=False)