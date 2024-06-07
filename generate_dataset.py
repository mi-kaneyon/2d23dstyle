import cv2
import numpy as np

# キャリブレーションデータの読み込み
calib_data_0 = np.load('calibration_data_0.npz')
calib_data_2 = np.load('calibration_data_2.npz')

mtx0 = calib_data_0['mtx']
dist0 = calib_data_0['dist']
mtx2 = calib_data_2['mtx']
dist2 = calib_data_2['dist']

# データ収集用のカメラセットアップ
cap0 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

if not cap0.isOpened():
    print("カメラ0を開けませんでした")
    exit()
if not cap2.isOpened():
    print("カメラ2を開けませんでした")
    exit()

# 収集したデータを格納するリスト
image_pairs = []
disparity_maps = []

# 明るさ調整パラメータ
alpha = 1.5  # コントラストの増加
beta = 50    # 明るさの増加

# 縮小パラメータ
resize_factor_0 = 0.9

while len(image_pairs) < 100:  # 100ペアのデータを収集
    ret0, frame0 = cap0.read()
    ret2, frame2 = cap2.read()

    if not ret0 or not ret2:
        print("カメラから映像を取得できませんでした")
        break

    # カメラ0の映像を縮小
    height0, width0, _ = frame0.shape
    frame0 = cv2.resize(frame0, (int(width0 * resize_factor_0), int(height0 * resize_factor_0)))

    # カメラ2の映像の明るさ調整
    frame2 = cv2.convertScaleAbs(frame2, alpha=alpha, beta=beta)

    # カメラ2の画像の中央部分をトリミング
    height0, width0, _ = frame0.shape
    height2, width2, _ = frame2.shape
    scale_factor = 1.2  # 拡大スケールを調整

    center_x, center_y = width2 // 2, height2 // 2
    half_width, half_height = int(width0 // (2 * scale_factor)), int(height0 // (2 * scale_factor))

    # トリミング範囲の調整
    left = max(center_x - half_width, 0)
    right = min(center_x + half_width, width2)
    top = max(center_y - half_height, 0)
    bottom = min(center_y + half_height, height2)
    cropped_frame2 = frame2[top:bottom, left:left + (right - left)]

    # トリミング後にリサイズ
    resized_frame2 = cv2.resize(cropped_frame2, (width0, height0))

    # ステレオマッチングを使用して視差マップを生成
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(resized_frame2, cv2.COLOR_BGR2GRAY)

    # サイズを一致させるためにリサイズ
    gray2 = cv2.resize(gray2, (gray0.shape[1], gray0.shape[0]))

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(gray0, gray2)

    # 画像ペアと視差マップをリストに追加
    image_pairs.append(np.stack([frame0, resized_frame2], axis=0))
    disparity_maps.append(disparity)

    # 進行状況を表示
    print(f'収集したデータペア数: {len(image_pairs)}')

cap0.release()
cap2.release()

# データをnumpy配列に変換して保存
image_pairs = np.array(image_pairs)
disparity_maps = np.array(disparity_maps)

# すべての画像ペアと視差マップのサイズを統一する
target_shape = (image_pairs.shape[0], 2, 432, 576, 3)
image_pairs_resized = np.zeros(target_shape, dtype=np.uint8)
disparity_maps_resized = np.zeros((image_pairs.shape[0], 432, 576), dtype=np.float32)

for i in range(image_pairs.shape[0]):
    for j in range(2):
        image_pairs_resized[i, j] = cv2.resize(image_pairs[i, j], (576, 432))
    disparity_maps_resized[i] = cv2.resize(disparity_maps[i], (576, 432))

np.save('image_pairs.npy', image_pairs_resized)
np.save('disparity_maps.npy', disparity_maps_resized)

print("データセットの生成が完了しました")
