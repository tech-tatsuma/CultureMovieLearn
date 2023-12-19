import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 動画ファイルを読み込む
cap = cv2.VideoCapture('./input_video.mp4')

# 動画のプロパティを取得
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# クロップ領域の指定
crop_x, crop_y, crop_w, crop_h = 500, 150, 200, 300  # 任意に調整してください
crop_x2, crop_y2, crop_w2, crop_h2 = 250, 150, 200, 300  # 任意に調整してください

# 輝度値を記録するためのリスト
mean_values = []
max_values = []
mean_values2 = []
max_values2 = []
bright_area = []
bright_area2 = []

# matplotlibのプロット用の準備
fig, ax = plt.subplots()
canvas = FigureCanvas(fig)

# プロット画像の出力サイズ
plot_width = width
plot_height = 300  # 任意に調整

# VideoWriterオブジェクトを作成(出力動画のサイズに注意)
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height + plot_height))

# 動画を読み込みながら処理
# while cap.isOpened():
while True:
    ret, frame = cap.read()
    if not ret:
        # continue
        break
    print(frame.shape)

    # クロップ領域のフレームを取得
    crop_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    crop_frame2 = frame[crop_y2:crop_y2 + crop_h2, crop_x2:crop_x2 + crop_w2]

    # add bounding box to crop_frame
    cv2.rectangle(crop_frame, (0, 0), (crop_w, crop_h), (0, 0, 255), 2)
    cv2.rectangle(crop_frame2, (0, 0), (crop_w2, crop_h2), (0, 255, 255), 2)

    # show crop frame
    # cv2.imshow('crop_frame', crop_frame)

    
    # クロップ画像の輝度値をグレースケールで計算
    gray_crop = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray_crop)
    max_val = np.max(gray_crop)

    gray_crop2 = cv2.cvtColor(crop_frame2, cv2.COLOR_BGR2GRAY)
    mean_val2 = np.mean(gray_crop2)
    max_val2 = np.max(gray_crop2)
    
    # 輝度値をリストに追加
    mean_values.append(mean_val)
    max_values.append(max_val)

    mean_values2.append(mean_val2)
    max_values2.append(max_val2)
    # 輝度が230以上の画素数をカウント
    bright_area.append(np.sum(gray_crop >= 230))
    bright_area2.append(np.sum(gray_crop2 >= 230))


    # if mean_val > 140:
    if np.sum(gray_crop >= 230) > 10000:
        cv2.putText(frame, 'OPEN!!: {:.2f}'.format(mean_val), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2)
    else:
        cv2.putText(frame, 'CLOSE: {:.2f}'.format(mean_val), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness=2)

    if np.sum(gray_crop2 >= 230) > 20000:
        cv2.putText(frame, 'Slab on line'.format(mean_val2), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), thickness=2)
    else:
        cv2.putText(frame, 'Slab off line'.format(mean_val2), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), thickness=2)
    
    # プロットデータの更新
    ax.clear()
    ax.plot(mean_values, label='Door Mean Luminance')
    # ax.plot(max_values, label='Max Luminance')
    ax.plot(mean_values2, label='Line Mean Luminance')
    ax.plot(bright_area, label='Door Bright Area')
    ax.plot(bright_area2, label='Line Bright Area')
    ax.legend()
    
    # プロットをキャンバスに描画し、画像として取得
    canvas.draw()
    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
    
    # プロット画像のリサイズ (出力動画の幅に合わせる)
    plot_image_resized = cv2.resize(plot_image, (plot_width, plot_height))
    combined_frame = np.vstack((frame, plot_image_resized))

    cv2.imshow('combined_frame', combined_frame)

    # 結合したフレームを出力動画に追加
    out.write(combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()

plt.close(fig)  # matplotlibの図を閉じる.vstack((frame, plot_image_resized))

