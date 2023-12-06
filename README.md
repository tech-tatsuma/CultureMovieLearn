# SlowFastNetworksを使った動画分類
## 学習環境の構築
### 学習環境の立ち上げ
```
sudo docker build -t culturevideolearn .
```
### 学習環境の開始
```
sudo docker run --rm -it --gpus all -v $PWD:/culturevideolearn culturevideolearn:latest
```

## Requirements

- Python 3
- moviepy: 動画の読み込みと分割に使用
- pandas: データ操作とCSVファイルの作成に使用
- argparse: コマンドライン引数の処理に使用
## 実行方法
### データの作成
```
python create_data.py --video_path [動画ファイルへのパス] --timeline_csv_path [タイムラインCSVファイルへのパス] --output_folder [出力ディレクトリへのパス]
```
このスクリプトは動画を１秒間隔のセグメントに分割し、それらにラベルをつけてcsvファイルに出力するために使用される。

- --video_path: 処理する動画ファイルへのパス
- --timeline_csv_path: 動画の各セグメントに対応するラベルが記述されたCSVファイルへのパス
- --output_folder: 分割された動画ファイルとラベル付けされたCSVを保存するディレクトリへのパス
### データの学習
```
python train.py --data [csvファイルパス] --epochs [学習回数] --patience [早期終了パラメータ] --batch [バッチサイズ] --lr [学習率] --seed [ランダムシード] --lr_search [学習率探索]
```

- --data: create_data.pyの実行により作成されたcsvファイルのパスを指定する
- --patience: validation lossの値がpatience回連続で下がらなかった場合にプログラムは早期終了する
- --lr_search: 最適な学習率が分からない場合はこのパラメータを'true'にする
### テスト
```
python test.py --csv_file [テストに利用するcsvのパス] --model_path [モデルのパス] --output_csv [出力するcsvファイルのパス] 
```

### Note
- タイムラインCSVファイルは、各行に時間,ラベルの形式で記述されている必要がある。
- 出力されるCSVファイルには、各セグメントのファイルパスと対応するラベルが含まれる。
## Model
### CustomSlowFastネットワーク
このネットワークは、動画分類のためのSlowFastネットワークアーキテクチャを使用したカスタムニューラルネットワーク。SlowFastモデルは動画内の早い動きと遅い動きの両方を捉えることができるため、動画内の物体の動きに敏感なネットワークとなっている。