import cv2
import numpy as np

# カメラの初期化
cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

if not cap0.isOpened():
    print("カメラ0を開けませんでした")
    exit()
if not cap2.isOpened():
    print("カメラ2を開けませんでした")
    exit()

# 明るさ調整パラメータ
alpha = 1.5  # コントラストの増加
beta = 50    # 明るさの増加

# カメラからフレームを取得
ret0, frame0 = cap0.read()
ret2, frame2 = cap2.read()

if not ret0 or not ret2:
    print("カメラから映像を取得できませんでした")
    exit()

cap0.release()
cap2.release()

# カメラ2の映像の明るさ調整
frame2 = cv2.convertScaleAbs(frame2, alpha=alpha, beta=beta)

# カメラ2の画像のトリミングとリサイズ
height0, width0, _ = frame0.shape
height2, width2, _ = frame2.shape
scale_factor = 1.6

center_x, center_y = width2 // 2, height2 // 2
half_width, half_height = int(width0 // (2 * scale_factor)), int(height0 // (2 * scale_factor))

left = max(center_x - half_width, 0)
right = min(center_x + half_width, width2)
top = max(center_y - half_height, 0)
bottom = min(center_y + half_height, height2)
cropped_frame2 = frame2[top:bottom, left:left + (right - left)]

resized_frame2 = cv2.resize(cropped_frame2, (width0, height0))

# 透明度を持たせて重ね合わせ
overlay_alpha = 0.5
overlay_frame = cv2.addWeighted(frame0, 1 - overlay_alpha, resized_frame2, overlay_alpha, 0)

cv2.imshow('Overlay', overlay_frame)
cv2.waitKey(5000)  # 表示を5秒間待機
cv2.destroyAllWindows()

# キャリブレーションのための仮のデータ
objpoints = [np.zeros((6*7, 3), np.float32) for _ in range(10)]
imgpoints0 = [np.random.rand(42, 1, 2).astype(np.float32) for _ in range(10)]
imgpoints2 = [np.random.rand(42, 1, 2).astype(np.float32) for _ in range(10)]

# 仮のキャリブレーションデータを使用
mtx0 = np.array([[640, 0, width0 / 2],
                 [0, 640, height0 / 2],
                 [0, 0, 1]])
dist0 = np.zeros(5)

mtx2 = np.array([[640, 0, width0 / 2],
                 [0, 640, height0 / 2],
                 [0, 0, 1]])
dist2 = np.zeros(5)

# 仮のキャリブレーションデータの保存
np.savez('calibration_data_0.npz', mtx=mtx0, dist=dist0)
np.savez('calibration_data_2.npz', mtx=mtx2, dist=dist2)

print("キャリブレーションが完了しました")
