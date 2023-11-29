# 学習環境の構築
## 学習環境の立ち上げ
```
sudo docker build -t culturevideolearn .
```
## 学習環境の開始
```
sudo docker run --rm -it --gpus all -v $PWD:/culturevideolearn culturevideolearn:latest
```