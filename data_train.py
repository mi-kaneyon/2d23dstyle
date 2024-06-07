import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

class DepthEstimationDataset(Dataset):
    def __init__(self, image_pairs, depth_maps):
        self.image_pairs = image_pairs
        self.depth_maps = depth_maps

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        image_pair = self.image_pairs[idx]
        depth_map = self.depth_maps[idx]
        image_pair = torch.tensor(image_pair, dtype=torch.float32).permute(0, 3, 1, 2)  # データをfloat32に変換し、形状を変換
        depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)  # データをfloat32に変換し、チャンネル次元を追加
        return image_pair, depth_map

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 出力を0-1の範囲に収める
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# データの読み込み
image_pairs = np.load('image_pairs.npy')
disparity_maps = np.load('disparity_maps.npy')
dataset = DepthEstimationDataset(image_pairs, disparity_maps)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# モデルの定義とGPUへの移行
model = DepthEstimationModel().to(device)
criterion = nn.SmoothL1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(300):  # エポック数を30に設定
    epoch_loss = 0
    for image_pair, depth_map in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        image_pair = image_pair.to(device)
        depth_map = depth_map.to(device)
        optimizer.zero_grad()

        # デバッグ用に形状を表示
        print(f'Original image_pair shape: {image_pair.shape}')
        
        try:
            # (batch_size, 2, 3, 432, 576) -> (batch_size, 6, 432, 576)
            image_pair = image_pair.reshape(image_pair.size(0), 6, image_pair.size(3), image_pair.size(4))
            print(f'Reshaped image_pair shape: {image_pair.shape}')
            print(f'depth_map shape: {depth_map.shape}')
            
            outputs = model(image_pair)
            loss = criterion(outputs, depth_map)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        except RuntimeError as e:
            print(f'Error during processing batch: {e}')
            print(f'image_pair shape: {image_pair.shape}, depth_map shape: {depth_map.shape}')
            continue

    scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

# 学習したモデルの保存
torch.save(model.state_dict(), 'depth_estimation_model.pth')
