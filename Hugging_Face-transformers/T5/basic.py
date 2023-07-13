from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_sequence_1 = 'Transfer learning, where a model is first pre-trained on a data-rich task before being finetuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.1 Keywords: transfer learning, natural language processing, multi-task learning, attentionbased models, deep learning'

task_prefix = 'summarize: '

input_sequences = [input_sequence_1]
# 入力データが複数ある場合（バッチ処理）は，ここで1つのリスト形式にしておく．
# ex) input_sequences = [input_sequence_1, input_sequence_2 ...]

encoding = tokenizer([task_prefix + sequence for sequence in input_sequences], padding='longest', max_length=512, truncation=True, return_tensors='pt')
input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

output = model.generate(input_ids=input_ids, max_new_tokens=128)
output_text = tokenizer.decode(output[0])
print("output_text : ", output_text)


"以下 non-autoregressive に推論させる方法（教師データ必須）"
# chat-gpt の要約結果を教師とする
label = 'Transfer learning in NLP involves pre-training models on data-rich tasks and fine-tuning for downstream tasks. This paper introduces a unified text-to-text framework, comparing objectives, architectures, datasets, and transfer approaches. Using the "Colossal Clean Crawled Corpus," they achieve state-of-the-art results in summarization, QA, and classification. Released dataset, models, and code support future NLP transfer learning. Keywords: transfer learning, NLP, multi-task learning, attention-based models, deep learning. '
labels = [label]
target_encoding = tokenizer(labels, padding = "longest", max_length = 128, truncation = True, return_tensors = "pt")

labels_id = target_encoding.input_ids

labels_id[labels_id == tokenizer.pad_token_id] = -100

output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_id)

print('loss : ',output.loss.item())
print('output_ids : ', output.logits) #print('output_ids : ', output['logits'])

ids = output['logits'].argmax(dim=2).numpy()
print('output_ids : ', ids[0])

tokens = tokenizer.convert_ids_to_tokens(ids[0])
#print("tokens : ", tokens)

string = tokenizer.convert_tokens_to_string(tokens)
print('output_string : ', string)

'''
実行結果：
output_text :  <pad> transfer learning is a powerful technique in natural language processing. the effectiveness of transfer learning has given rise to a diversity of approaches, methodologies, and practice. in this paper, we explore the landscape of transfer learning techniques for NLP.</s>
loss :  3.7194948196411133
output_ids :  tensor([[[-15.6880,  -8.7997, -12.4212,  ..., -38.4968, -38.6359, -38.5825],
         [-31.4625, -13.8503, -16.9880,  ..., -50.5683, -50.6882, -50.6803],
         [-26.9660, -10.6865, -12.4944,  ..., -42.8414, -42.9576, -43.0797],
         ...,
         [-42.8160, -10.6591, -21.9949,  ..., -60.8817, -60.9876, -60.9620],
         [-49.0436,  -5.0765, -22.0306,  ..., -62.8554, -62.9858, -62.9818],
         [-55.6809,  -6.2772, -24.7481,  ..., -66.4583, -66.5662, -66.5050]]],
       grad_fn=<UnsafeViewBackward0>)
output_ids :  [ 2025  1036    19   793  6892    19   554    18    17  7233    11   331
    18  3723  4145     3  1399    17    17   444    53 26804 26804  4145
     3     3  1040  4048     7     3     9  4732 22927  4732    18   235
    18  6327  4732    24    84 13275   554     6  4648     7     6    73
     7     6  2025   119  6315     3     1     5     8  7639  3881  2298
     7   138  7433   205 10936  1361 10052   302   121    62  1984   538
    18   858    18   532    18  1408   772    30   186  1635  1707     6
   822     5     6  1499  1499     5     1    26   331     7   554     6
    11  1081     5     8   161  6892   161  1036  2097     1     7    10
  2025  1036     6   793  6892     1     1     1     1     1     1     1
     1     1     1     1     1     1     1     1]
output_string :  transfer learning is naturalLP is pre-t objectives and data-rich tasks  finettuneing downstream downstream tasks   paper compares a frameworkunified framework-to-text framework that whichcombining pre, architectures, uns, transfer other approaches </s> . the insightsColossal Clean Crawled Corpus" we achieve state-of-the-art results on manymarization, question., text text.</s> d datas pre, and code. the workLP work learning techniques</s> s: transfer learning, naturalLP</s></s></s></s></s></s></s></s></s></s></s></s></s></s></s>
'''
