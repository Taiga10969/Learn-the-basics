# LLaMA
Metaが公開したLLaMAのモデルについて

## 学習済.pthファイルについて
Google Formへのリンクから申請する必要がある．（省略）
個人所有の```llama.tar.gz``` ファイルの中身は，.pth ファイルのみ格納されたファイルである．
別途，```tokenizer.model```，```tokenizer_checklist.chk``` が格納された```.tar.gz ```ファイルを解凍し，以下のファイル構造になるように移動させる．
※厳密には，実行時にパスを指定すればいいと思うが，以下のファイル構造を構築する．

```
F:\LLAMA
│  tokenizer.model
│  tokenizer_checklist.chk
│
├─7B
│      consolidated.00.pth
│      params.json
│      checklist.chk
│
├─13B
│      consolidated.00.pth
│      consolidated.01.pth
│      params.json
│      checklist.chk
│
├─30B
│      consolidated.00.pth
│      consolidated.01.pth
│      consolidated.02.pth
│      consolidated.03.pth
│      params.json
│      checklist.chk
│
└─65B
        consolidated.00.pth
        consolidated.01.pth
        consolidated.02.pth
        consolidated.03.pth
        consolidated.04.pth
        consolidated.05.pth
        consolidated.06.pth
        consolidated.07.pth
        checklist.chk
        params.json

```
 

## 動作確認
LLaMAを動かすまでの過程を以下に示す．

[参考]
https://github.com/facebookresearch/llama
https://qiita.com/lyulu/items/59e1633c3184963d1a59

環境構築
```
pip install pytorch torchvision torchaudio
pip install -r requirements.txt
pip install -e .
```

github上のLLaMAのディレクトリを引っ張ってくる．
```git clone https://github.com/facebookresearch/llama.git```


