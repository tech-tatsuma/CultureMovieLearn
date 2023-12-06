# ビデオ予測API
１秒間のビデオファイルをリクエストとして受け取り、学習済みモデルを使用して予測を行うFastAPIベースのAPIを実装したプロジェクト
## Model
### CustomSlowFastネットワーク
このネットワークは、動画分類のためのSlowFastネットワークアーキテクチャを使用したカスタムニューラルネットワーク。SlowFastモデルは動画内の早い動きと遅い動きの両方を捉えることができるため、動画内の物体の動きに敏感なネットワークとなっている。
## APIの立ち上げ
```
python router.py
```
デフォルトでは、APIはhttp://127.0.0.1:8000 で立ち上がるようになっているが、独自の環境に合わせて利用する。
## リクエスト方法
predict-videoエンドポイント

- メソッド：POST
- 説明：このエンドポイントは、ビデオファイルを受け取り、ビデオに対する予測を行う
- リクエスト形式：フォームデータでビデオファイルをアップロードする
### リクエスト例
```
curl -X 'POST' \
  'http://127.0.0.1:8000/predict-video' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path_to_video_file'
```
### レスポンス形状
- current: 現在のフレームの予測結果
- next: 次のフレームの予測結果
```
{
  "current": true,
  "next": false
}
```
- true：給餌
- false: 給餌停止

### 参考
- https://arxiv.org/abs/1812.03982
- https://github.com/facebookresearch/SlowFast