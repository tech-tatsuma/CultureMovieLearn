import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import csv
import argparse

# csvファイルのindex列目のデータをリストとして抽出する関数
def extract_column_from_csv_skip_header(csv_path, index):
    data = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) > index:
                append_data = row[index]
                if index == 1:
                    if row[index] == '0':
                        append_data = 0
                    else:
                        append_data = 1
                    data.append(float(append_data))
                data.append(row[index])
            else:
                data.append(None)
    return data

def create_movie(input_video, output_video, csv_file):

    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(input_video)

    # 動画のプロパティを取得
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 予測値と正解値を格納するためのリスト
    trues = extract_column_from_csv_skip_header(csv_file, 1)
    predictions = extract_column_from_csv_skip_header(csv_file, 2)

    # matplotlibのプロット用の準備
    fig, ax = plt.subplots()
    canvas = FigureCanvas(fig)

    # プロット画像の出力サイズ
    plot_width = width
    plot_height = 300

    # VideoWriterオブジェクトを作成
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height + plot_height))

    # 現在のフレーム番号
    current_frame = 0

    # ラベルとフレームの対応を計算するためのレート
    label_frame_rate = fps

    # 動画を読み込みながら処理
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # プロットデータの更新
        ax.clear()
        # 現在のラベルのインデックスを計算
        label_index = int(current_frame / label_frame_rate)
        if label_index < len(trues) and label_index < len(predictions):
            true_value = trues[label_index]
            prediction_value = predictions[label_index]
            
            # 真の値と予測値をプロット
            ax.plot(range(label_index + 1), trues[:label_index + 1], label='True Labels')
            ax.plot(range(label_index + 1), predictions[:label_index + 1], label='Predicted Labels')

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

        # 次のフレームに進む
        current_frame += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create movie from video and csv file')
    parser.add_argument('--input_video', type=str, help='Input video file path')
    parser.add_argument('--output_video', type=str, help='Output video file path')
    parser.add_argument('--csv_file', type=str, help='CSV file path')
    args = parser.parse_args()

    create_movie(args.input_video, args.output_video, args.csv_file)