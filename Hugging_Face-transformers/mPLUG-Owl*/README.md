# mPLUG-owl

github : https://github.com/X-PLUG/mPLUG-Owl

※ transformersの中に直接収録されているモデルではないが，transofrmersのコード記述方式で実装可能．

## 環境構築　（まあまあ苦労した...）

#### ● Docker image (Python3.10のみ，cuda12.10，ubuntu22.04)
taiga10969/basic_image:cuda12.1.0-ubuntu22.04-python3.10<br>
https://hub.docker.com/r/taiga10969/basic_image/tags<br>

#### ● Docker image　（mPLUG-Owl動作用，cuda12.10，ubuntu22.04）
taiga10969/mplug-owl:cuda12.10ubuntu22.04<br>
https://hub.docker.com/r/taiga10969/mplug-owl<br>

#### 環境
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
pip install -U protobuf==3.20.1
```

