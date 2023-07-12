# GPT2
GPT2について色々...<br>
<br>
`model` : non-autregressiveに次の単語を予測する<br>
`model.gererate` : autoregressiveに次単語予測を行う（推論）<br>
<br>
※`model.generate` ではモデルの学習は行えない．と解釈<br>


## practice.py
モデルの動作確認コード
### 必要ライブラリのインポート
```
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
```
### トークナイザーとモデルの定義
```
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print('tokenizer.vocab_size : ', tokenizer.vocab_size)
model = GPT2LMHeadModel.from_pretrained('gpt2')
```
