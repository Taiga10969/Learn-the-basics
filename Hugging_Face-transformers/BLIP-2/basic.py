from transformers import AutoProcessor, Blip2ForConditionalGeneration, AutoTokenizer
import torch
from PIL import Image

# モデルの定義
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


image1 = Image.open("/taiga/experiment/BLIP-2/sample_img.jpg")       #愛犬の写真
image2 = Image.open("/taiga/experiment/BLIP-2/sample_figure1.png")   #論文図1
image3 = Image.open("/taiga/experiment/BLIP-2/sample_figure2.png")   #論文図2

inputs = processor(image1, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("img : ", generated_text)

inputs = processor(image2, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("figure1 : ", generated_text)

inputs = processor(image3, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("figure2 : ", generated_text)


print('===== add prompt =====')

prompt = "The types of this diagram are"
print('prompt : ', prompt)

inputs = processor(image2, text=prompt, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("figure1 : ", generated_text)

inputs = processor(image3, text=prompt, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("figure2 : ", generated_text)


##
prompt = "Question: What type of diagram is this? Answer: "
inputs = processor(images=image2, text=prompt, return_tensors="pt").to(device, torch.float16)
print(inputs)

inputs = {k: v.to(device) for k, v in inputs.items()}

# ラベルのテンソル化
label = "this diagram type is chart"
label = processor(text=label, return_tensors="pt")["input_ids"]
label = label.to(device)

outputs = model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"], labels=label)

ids = outputs['logits'].argmax(dim=2).cpu().numpy()
print('output_ids : ', ids[0])

generated_text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

print("figure1 : ", generated_text)


tokens = tokenizer.convert_ids_to_tokens(ids[0])
print("tokens : ", tokens)

string = tokenizer.convert_tokens_to_string(tokens)
print('output_string : ', string)

'''出力結果
Loading checkpoint shards: 100%|██████████████████████| 2/2 [00:08<00:00,  4.32s/it]
figure1 :  a graph showing the different types of data
figure2 :  three different images of a river and a river
===== add prompt =====
prompt :  The types of this diagram are
figure1 :  shown in different colors
figure2 :  shown in different colors
{'pixel_values': tensor([[[[1.9307, 1.9307, 1.9307,  ..., 1.9307, 1.9307, 1.9307],
          [1.9307, 1.9307, 1.9307,  ..., 1.9307, 1.9307, 1.9307],
          [1.9307, 1.9307, 1.9307,  ..., 1.7695, 1.9307, 1.9307],
          ...,
          [1.9307, 1.9307, 1.9307,  ..., 1.9307, 1.9307, 1.9307],
          [1.9307, 1.9307, 1.9307,  ..., 1.9307, 1.9307, 1.9307],
          [1.9307, 1.9307, 1.9307,  ..., 1.9307, 1.9307, 1.9307]],

         [[2.0742, 2.0742, 2.0742,  ..., 2.0742, 2.0742, 2.0742],
          [2.0742, 2.0742, 2.0742,  ..., 2.0742, 2.0742, 2.0742],
          [2.0742, 2.0742, 2.0742,  ..., 1.9102, 2.0742, 2.0742],
          ...,
          [2.0742, 2.0742, 2.0742,  ..., 2.0742, 2.0742, 2.0742],
          [2.0742, 2.0742, 2.0742,  ..., 2.0742, 2.0742, 2.0742],
          [2.0742, 2.0742, 2.0742,  ..., 2.0742, 2.0742, 2.0742]],

         [[2.1465, 2.1465, 2.1465,  ..., 2.1465, 2.1465, 2.1465],
          [2.1465, 2.1465, 2.1465,  ..., 2.1465, 2.1465, 2.1465],
          [2.1465, 2.1465, 2.1465,  ..., 1.9893, 2.1465, 2.1465],
          ...,
          [2.1465, 2.1465, 2.1465,  ..., 2.1465, 2.1465, 2.1465],
          [2.1465, 2.1465, 2.1465,  ..., 2.1465, 2.1465, 2.1465],
          [2.1465, 2.1465, 2.1465,  ..., 2.1465, 2.1465, 2.1465]]]],
       device='cuda:0', dtype=torch.float16), 'input_ids': tensor([[    2, 45641,    35,   653,  1907,     9, 41071,    16,    42,   116,
         31652,    35,  1437]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}
output_ids :  [  275   116 50118    35    85 27203]
figure1 :  best?
: It_____
tokens :  ['Ġbest', '?', 'Ċ', ':', 'ĠIt', '_____']
output_string :   best?
: It_____

'''



