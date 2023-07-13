# Hugging_Face-transformers/T5
T5 (Text-to-Text Transfer Transformer) について色々

## ●T5 : Text-To-Text Transfer Transformer
Text-to-Text Transfer Transformer は分類，翻訳，要約といった様々な自然言語処理タスクを “Text-to-Text” で解くモデル．<br>
“Text-to-Text” とは入力を"タスク：問題"，出力を"回答"の形式として，全てのタスクを同じモデルで解いてしまおう！というもの．<br>
学習データだけ変えれば，同じモデルで様々なタスクを解くことができるという魅力がある．

### モデル構造
Encoder-Decoder, Language model, Prefix LM の比較<br>
※show the paper 3.2

## ●必要ライブラリ
```
pip install transformers
pip install SentencePiece
```
※実際にコードを記述する際に ```SentencePiece``` は import しないが，これをインストールしないと以下のエラーが表示される．<br>
　pip でインストールすることで解決した．
```
Traceback (most recent call last):
  File "practice.py", line 4, in <module>
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1026, in __getattribute__
    requires_backends(cls, cls._backends)
  File "/usr/local/lib/python3.8/dist-packages/transformers/utils/import_utils.py", line 1014, in requires_backends
    raise ImportError("".join(failed))
ImportError: 
T5Tokenizer requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
```
以上の環境のDocker iamge : <br>


## ●basic.py
T5の動作確認を行う．<br>
今回は，T5の論文のアブストラクトを入力し，```task_prefix = 'summarize: '``` として，アブストラクトを要約するタスクを行う．<br>
### tokenizer と model の定義
```
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
```
### 入力データ と タスクプレフィックスの用意 / データのID化
入力データは，T5の論文のアブストラクト全文をテキストデータとして入力し，タスクは，'summarize: ' として，これをプレフィックスとして1つの入力データとする．また，入力データのテキストデータを数値 (ID) データに変換する．
```
input_sequence_1 = 'Transfer learning, where a model is first pre-trained on a data-rich task before being finetuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.1 Keywords: transfer learning, natural language processing, multi-task learning, attentionbased models, deep learning'

task_prefix = 'summarize: '

input_sequences = [input_sequence_1]
# 入力データが複数ある場合（バッチ処理）は，ここで1つのリスト形式にしておく．
# ex) input_sequences = [input_sequence_1, input_sequence_2 ...]

encoding = tokenizer([task_prefix + sequence for sequence in input_sequences], padding='longest', max_length=512, truncation=True, return_tensors='pt')
input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
```
### 推論
```
output = model.generate(input_ids=input_ids, max_new_tokens=128)
output_text = tokenizer.decode(output[0])
print("output_text : ", output_text)
```



## 【参考】<br>
[T5 paper]<br>
https://arxiv.org/abs/1910.10683<br>
[huggingface document]<br>
https://huggingface.co/docs/transformers/model_doc/t5<br>
[Macでサイバーエージェントの公開したLLMを動かしてみる]<br>
https://llcc.hatenablog.com/entry/2023/05/20/225822<br>
[はじめての自然言語処理 第7回 T5 によるテキスト生成の検証]<br>
https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part7.html#fn1
