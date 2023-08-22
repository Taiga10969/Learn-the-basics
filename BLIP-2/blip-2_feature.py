#from imagenet_zeroshot_data import openai_imagenet_template, imagenet_classnames
import os
import glob
import torch
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm
import numpy as np



# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')


model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2", model_type="pretrain", is_eval=True, device=device)
model.visual_encoder.float()

prompts = ['Question: which city is this? Answer:']
files = ['/taiga/experiment/BLIP-2/image/merlion.png']


#with torch.no_grad():
#    for prompt in tqdm(prompts):
#        output = txt_processors["eval"](prompt)
#        print(output) 

text_embeddings = []
with torch.no_grad():
    for item in tqdm(prompts):
        text = model.tokenizer(
            item,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        print(text)
        text_feat = model.forward_text(text)
        print(text_feat)
        text_embed = F.normalize(model.text_proj(text_feat))
        print(text_embed)
        text_embeddings.append(text_embed.mean(dim=0, keepdim=True))
    text_embeddings = torch.cat(text_embeddings, dim=0)
    print(text_embeddings.shape) # torch.Size([1, 256])



image_embeddings = []
with torch.no_grad():
    for f in tqdm(files):
        raw_image = Image.open(f).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)
        image_embeddings.append(image_embed)
image_embeddings = torch.cat(image_embeddings, dim=0)
print(image_embeddings.shape) # torch.Size([1, 32, 256])

