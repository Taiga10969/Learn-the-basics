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
### model()
model()にID化した単語を入力すると各単語における次単語の予測を行う．<br>
```tokkenizer()``` の返却値は，'input_ids'と'attention_mask'が格納された辞書型のデータである．<br>
model()の```input_ids```への入力としては，`tokenizer` の出力のうち 'input_ids' の部分のみを入力する．<br>
また，model()の引数として，```labels``` に ```input_ids``` に入力した同じテキストを入力することで，損失値の計算が可能．<br>
また，model()の返却値としては，'loss'，'logits'，'past_key_values'，'hidden_states'，'attentions'，'cross_attention' が格納された辞書型のデータである．

```
text = "My name is Taiga , I like baseball"

input = tokenizer(text, return_tensors='pt')
print('input : ', input)

with torch.no_grad():
    outputs = model(input_ids = input['input_ids'], labels = input['input_ids'])

print('loss : ', outputs.loss)
print('outputs[logits].shape : ', outputs['logits'].shape)
```

### generate
generatedでは，ある単語列を入力した場合，Autoregressiveにその後の単語列を予測するというもの．
```
text = "My name is Taiga ,"
input = tokenizer(text, return_tensors='pt')
print('input : ', input)
output = model.generate(inputs=input['input_ids'], max_length=32, min_length=5, pad_token_id=50256, eos_token_id=50256, early_stopping=True)
string = tokenizer.decode(output[0])
print(string)
```

### 参考
https://gotutiyan.hatenablog.com/entry/2022/02/23/133414<br>
https://www.youtube.com/watch?v=elUCn_TFdQc
