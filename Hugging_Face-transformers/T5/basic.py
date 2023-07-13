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
