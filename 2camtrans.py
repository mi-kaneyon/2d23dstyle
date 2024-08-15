import cv2
import torch
import torch.nn as nn
import numpy as np

class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        return x

# モデルの初期化
model = FeatureExtractorCNN()

# カメラの初期化
cap1 = cv2.VideoCapture(0)  # 正面カメラ
cap2 = cv2.VideoCapture(2)  # 側面カメラ

# カメラが開けたかの確認
if not cap1.isOpened():
    print("カメラ1を開けませんでした")
    exit()
if not cap2.isOpened():
    print("カメラ2を開けませんでした")
    exit()

# 解像度の設定
width, height = 640, 480  # カメラの解像度
target_size = (128, 128)  # モデルの入力サイズ

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# ウィンドウを自由に調整できるように設定
cv2.namedWindow('Combined Heatmap', cv2.WINDOW_NORMAL)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("カメラ映像の取得に失敗しました")
        break

    # フレームのリサイズ
    frame1_resized = cv2.resize(frame1, target_size)
    frame2_resized = cv2.resize(frame2, target_size)

    # OpenCVからPyTorchテンソルへの変換
    frame1_tensor = torch.tensor(frame1_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    frame2_tensor = torch.tensor(frame2_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 特徴量マップの抽出
    with torch.no_grad():
        feature_map1 = model(frame1_tensor)
        feature_map2 = model(frame2_tensor)

    # 各チャンネルの平均を取って1チャンネルに変換
    feature_map1 = feature_map1.mean(dim=1).squeeze().numpy()
    feature_map2 = feature_map2.mean(dim=1).squeeze().numpy()

    # 特徴量マップを適切なフォーマットに変換（8ビット整数型）
    feature_map1_normalized = cv2.normalize(feature_map1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    feature_map2_normalized = cv2.normalize(feature_map2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # ヒートマップに変換（グレースケールからカラーマップへ）
    heatmap1 = cv2.applyColorMap(feature_map1_normalized, cv2.COLORMAP_JET)
    heatmap2 = cv2.applyColorMap(feature_map2_normalized, cv2.COLORMAP_JET)

    # ヒートマップをウィンドウに表示するためにリサイズ
    heatmap1_resized = cv2.resize(heatmap1, (640, 480))
    heatmap2_resized = cv2.resize(heatmap2, (640, 480))

    # アルファブレンドで2つのヒートマップを透過して重ねる
    alpha = 0.5  # 透過度の割合
    combined_heatmap = cv2.addWeighted(heatmap1_resized, alpha, heatmap2_resized, 1 - alpha, 0)

    # 結果のリアルタイム表示
    cv2.imshow('Combined Heatmap', combined_heatmap)

    # 'q' キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
