from moviepy.editor import VideoFileClip
import os
import pandas as pd
import argparse

# 動画を１秒ごとに分割する関数
def split_video(video_path, output_folder):
    # 動画を読み込む
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)

    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    outputs = []

    # 1秒ごとに動画を分割して保存
    for i in range(duration-1):
        # 分割する動画の時間範囲を設定
        subclip = clip.subclip(i, i+1)

        # 分割した動画をファイルとして保存（フレームレートは25fpsに設定）
        output_path = os.path.join(output_folder, f"segment_{i}.mp4")
        subclip.write_videofile(output_path, codec="libx264", fps=25)

        outputs.append(output_path)

    clip.close()

    return outputs

def create_labeled_video_csv(video_paths, csv_content):
    # CSVファイルの内容を解析して時間範囲とステータスを辞書に保存
    time_ranges = csv_content.split('\n')
    status_dict = {}
    prev_time = 0
    for range_info in time_ranges:
        if not range_info.strip():
            continue
        time, status = map(int, range_info.split(','))
        status_dict[range(prev_time, time)] = status
        prev_time = time
    status_dict[range(prev_time, float('inf'))] = 0

    # 各動画セグメントにラベルを割り当て
    labeled_data = []
    for video_path in video_paths:
        # ファイル名から時間を抽出
        segment_time = int(video_path.split('_')[-1].split('.')[0])
        # 適切なステータスを見つける
        label = next(status for time_range, status in status_dict.items() if segment_time in time_range)
        labeled_data.append((video_path, label))

    # DataFrameを作成し、CSVファイルに保存
    df = pd.DataFrame(labeled_data, columns=['video_path', 'status'])
    return df

def read_csv_to_string(csv_path):
    # CSVファイルを読み込む
    with open(csv_path, 'r') as file:
        content = file.read()
    # 改行で連結された文字列として返す
    return content.strip()

def write_dataframe_to_csv(df, output_path):
    # DataFrameをCSVファイルとして書き込む
    df.to_csv(output_path, index=False)

def main(video_path, timeline_csv_path, output_folder):
    # 動画を分割して出力フォルダに保存
    video_segments = split_video(video_path, output_folder)

    # タイムラインのCSVデータを読み込む
    timeline_data = read_csv_to_string(timeline_csv_path)

    # ラベル付けされた動画データのDataFrameを作成
    labeled_df = create_labeled_video_csv(video_segments, timeline_data)

    # 出力フォルダの親ディレクトリを取得
    parent_folder = os.path.dirname(output_folder)

    # DataFrameの1列目（動画パス）を出力フォルダの親ディレクトリに対する相対パスに変換
    labeled_df['video_path'] = labeled_df['video_path'].apply(
        lambda x: os.path.relpath(x, parent_folder)
    )

    # 結果をCSVファイルとして出力フォルダの親ディレクトリに書き込む
    output_csv_path = os.path.join(parent_folder, 'labeled_videos.csv')
    write_dataframe_to_csv(labeled_df, output_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('timeline_csv_path', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()

    main(args.video_path, args.timeline_csv_path, args.output_folder)