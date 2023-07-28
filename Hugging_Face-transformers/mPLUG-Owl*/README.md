# mPLUG-owl

github : https://github.com/X-PLUG/mPLUG-Owl

※ transformersの中に直接収録されているモデルではないが，transofrmersのコード記述方式で実装可能．

## 環境構築　（まあまあ苦労した...）
Docker image : https://hub.docker.com/r/taiga10969/mplug-owl<br>
環境
* python==3.10
* torch==1.13.1
* torchvision==0.14.1
* torchaudio==0.13.1
```
apt-get install python3-dev
```
```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```
```
pip install -r requirements.txt
```


エラー解消のために…
```
pip install protobuf==3.20.1
```

