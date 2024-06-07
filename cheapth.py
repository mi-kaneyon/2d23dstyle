import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms

# 学習したモデルの定義とロード
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

# 学習したモデルのロード
model = DepthEstimationModel().to(device)
model.load_state_dict(torch.load('depth_estimation_model.pth'))
model.eval()

# ビデオキャプチャの初期化
cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

# カメラキャリブレーションデータのロード
cal_data_0 = np.load('calibration_data_0.npz')
cal_data_2 = np.load('calibration_data_2.npz')
mtx0 = cal_data_0['mtx']
dist0 = cal_data_0['dist']
mtx2 = cal_data_2['mtx']
dist2 = cal_data_2['dist']

while True:
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()

    if not ret0 or not ret2:
        break

    # 画像の補正
    frame0 = cv2.undistort(frame0, mtx0, dist0)
    frame2 = cv2.undistort(frame2, mtx2, dist2)

    # カメラ0の映像をクロップしてカメラ2に合わせる
    h0, w0 = frame0.shape[:2]
    frame0_cropped = frame0[:, int(w0 * 0.2):int(w0 * 0.8)]  # 60%の画角にクロップ

    # リサイズしてカメラ2のサイズに合わせる
    frame0_resized = cv2.resize(frame0_cropped, (frame2.shape[1], frame2.shape[0]))

    # カメラ2とカメラ0の画像を結合
    combined_image = np.concatenate((frame2, frame0_resized), axis=2)  # チャンネル軸で結合

    # 画像をTensorに変換
    input_image = transforms.ToTensor()(combined_image).unsqueeze(0).to(device)

    # デプス推定
    with torch.no_grad():
        depth_map = model(input_image).squeeze().cpu().numpy()

    # デプスマップをカラーマップに変換
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # 結果の表示（デプスマップのみを表示）
    cv2.imshow('Depth Map', depth_map_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap2.release()
cv2.destroyAllWindows()
