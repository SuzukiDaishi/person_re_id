"""動画をフレームごとにセグメントに分割するスクリプト"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 入力動画ファイルのパス
video_path = 'dataset_youtube/思い出の交差点 short film プロローグ【短編映画】.mp4'
# 出力動画ファイルのテンプレート（セグメントごとに番号付き）
output_template = 'segment_{}.mp4'
# 差分の閾値
threshold = 35

# 動画の読み込み
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("動画を読み込めませんでした。")
    exit()

# 動画情報の取得（FPS, サイズなど）
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'avc1')

# セグメント番号の初期化
segment_counter = 1
# 最初のセグメント用の VideoWriter を作成
out = cv2.VideoWriter(output_template.format(segment_counter), fourcc, fps, (width, height))

# 最初のフレームを取得
ret, prev_frame = cap.read()
if not ret:
    print("最初のフレームを取得できませんでした。")
    cap.release()
    exit()

# 最初のフレームはグレースケールに変換して基準とする
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# 最初のフレームを書き込み
out.write(prev_frame)

# 差分値の平均値を記録するリスト（グラフ用）
diff_means = []
frame_count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 前フレームとの絶対差分を計算
    frame_diff = cv2.absdiff(prev_gray, gray)
    mean_diff = np.mean(frame_diff)
    diff_means.append(mean_diff)

    # 差分が閾値以上の場合はセグメントを切る
    if mean_diff >= threshold:
        print(f"Frame {frame_count}: 差分値 {mean_diff:.2f} が閾値 {threshold} を超えたため、セグメントを分割します。")
        # 現在のセグメントの VideoWriter を解放
        out.release()
        # 新たなセグメント番号を更新し、新しい VideoWriter を作成
        segment_counter += 1
        out = cv2.VideoWriter(output_template.format(segment_counter), fourcc, fps, (width, height))
        # 切り替えたフレームを新セグメントに書き込む
        out.write(frame)
    else:
        # 閾値未満の場合は現在のセグメントにフレームを書き込み
        out.write(frame)

    # 比較用に現在のフレームを保持
    prev_gray = gray
    frame_count += 1

# リソースの解放
cap.release()
out.release()

# 差分値のグラフを描画
plt.figure(figsize=(10, 5))
plt.plot(diff_means, label='Average Color Difference')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
plt.xlabel('Frame')
plt.ylabel('Difference')
plt.title('Frame-to-Frame Color Difference')
plt.legend()
plt.show()
