# BLIP-2


## 必要ライブラリのインポート
```
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

#別途画像を読み込むのに必要
from PIL import Image
```

## モデルの定義
```
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
```

## 推論
Autoregressiveに推論させる．
```
inputs = processor(image1, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("img : ", generated_text)
```

## 推論2
Non-Autoregressiveに推論させる．
```
prompt = "Question: What type of diagram is this? Answer:"
inputs = processor(images=image1, text=prompt, return_tensors="pt").to(device, torch.float16)
print(inputs)

inputs = {k: v.to(device) for k, v in inputs.items()}

# ラベルのテンソル化
label = "this diagram type is chart"
label = processor(text=label, return_tensors="pt")["input_ids"]
label = label.to(device)

outputs = model(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"], labels=label)

print(outputs)
#output = model(**inputs)
attributes = dir(outputs)
print(attributes)
```
このoutputsから得られるものは以下のようになっている．<br>
```['__annotations__', '__class__', '__contains__', '__dataclass_fields__', '__dataclass_params__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__post_init__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'language_model_outputs', 'logits', 'loss', 'move_to_end', 'pop', 'popitem', 'qformer_outputs', 'setdefault', 'to_tuple', 'update', 'values', 'vision_outputs']```


## 参考
[paper] https://arxiv.org/abs/2301.12597 <br>
[github] https://github.com/huggingface/blog/blob/main/blip-2.md <br>
[Hugging Face] https://huggingface.co/docs/transformers/main/model_doc/blip-2 <br>

**その他参考になりそうなサイト**<br>
[[翻訳] Hugging Face transformersにおける前処理] https://qiita.com/taka_yayoi/items/d6300140765b9b1406c5<br>
[論文まとめ] https://blog.shikoan.com/blip-2/#%E4%BD%BF%E3%81%A3%E3%81%A6%E3%81%BF%E3%81%9F%E6%89%80%E6%84%9F
