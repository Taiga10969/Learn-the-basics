from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print('tokenizer.vocab_size : ', tokenizer.vocab_size)

model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "My name is Taiga , I like baseball"

input = tokenizer(text, return_tensors='pt')
print('input : ', input)

with torch.no_grad():
    outputs = model(input_ids = input['input_ids'], labels = input['input_ids'])

print('loss : ', outputs.loss)
print('outputs[logits].shape : ', outputs['logits'].shape)


ids = outputs['logits'].argmax(dim=2).numpy()
print('output_ids : ', ids[0])

tokens = tokenizer.convert_ids_to_tokens(ids[0])
print("tokens : ", tokens)

string = tokenizer.convert_tokens_to_string(tokens)
print('output_string : ', string)


#generate
text = "My name is Taiga ,"
input = tokenizer(text, return_tensors='pt')
print('input : ', input)
output = model.generate(inputs=input['input_ids'], max_length=32, min_length=5, pad_token_id=50256, eos_token_id=50256)
string = tokenizer.decode(output[0])
print(string)


#print(outputs['logits'].contiguous().shape)
#print(torch.exp(outputs.loss)) # tensor(312.8972)
#print('output[last_hidden_state].shape', output['last_hidden_state'].shape)
#output_text = tokenizer.decode(output['last_hidden_state'])
#print('output text : ', output_text)
#output_text = model.generate(text, max_length=100, early_stopping=True)
#print('output text : ', output_text)

## 参考
# https://gotutiyan.hatenablog.com/entry/2022/02/23/133414
# https://www.youtube.com/watch?v=elUCn_TFdQc
