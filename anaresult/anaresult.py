import cv2
import csv
import os

def write_values_on_video(video_path, value, status, true_data, output_path):
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 書き込み用のビデオライターを設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # テキストを書き込む
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Value: {value}', (width - 200, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Status: {status}', (width - 200, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'True: {true_data}', (width - 200, 90), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 書き込んだフレームを出力
        out.write(frame)

    # リソースを解放
    cap.release()
    out.release()

def concatenate_videos(video_paths, output_path):
    # 最初の動画を読み込んで、フレームサイズとフレームレートを取得
    cap = cv2.VideoCapture(video_paths[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 書き込み用のビデオライターを設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # フレームを出力動画に書き込む
            out.write(frame)

        cap.release()

    # 出力ファイルを閉じる
    out.release()

def process_csv_and_write_text(csv_path, final_output_path):
    csv_dir = os.path.dirname(csv_path)

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 最初の行（ヘッダー）をスキップ

        outputs = []

        for index, row in enumerate(reader):
            relative_video_path = row[0]
            video_path = os.path.join(csv_dir, relative_video_path)
            video_path = video_path.replace('\\', '/')
            try:
                prediction = float(row[3])
            except ValueError:
                # predictionの値が無効な場合はこの行の処理をスキップ
                continue
            status = 'Feeding' if prediction >= 0.5 else 'No Feeding'
            true_data = 'No Feeding' if row[1] == '0' else 'Feeding'  # ここを修正
            output_path = f'output_{index}.mp4'  # 出力ファイル名を生成

            write_values_on_video(video_path, prediction, status, true_data, output_path)
            outputs.append(output_path)

        concatenate_videos(outputs, final_output_path)

        # 一時動画ファイルを削除
        for output in outputs:
            os.remove(output)

if __name__ == '__main__':
    csv_path = '/data/furuya/side_datasets/side_result2_modified.csv'
    final_output_path = './sideoutput.mp4'
    process_csv_and_write_text(csv_path, final_output_path)

