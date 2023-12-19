import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

import utils
from detr_model import DETRdemo


# モデルを定義
model = DETRdemo(num_classes=91)

# transformを定義
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 学習済みパラメータを読み込む
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
msg = model.load_state_dict(state_dict)
print(f"load_state_dict[info] >> msg : {msg}")

# モデルを評価モードにする
model.eval()

# 画像データをリスト化
img_directory_path = './sample_img'
image_files = utils.get_file_names(img_directory_path)

# 画像データの読み込み
img_idx = 0 # image_filesのindexを指定
image = Image.open(os.path.join(img_directory_path, image_files[img_idx])).convert('RGB')

# 画像の前処理
img = transform(image).unsqueeze(0)
assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

# 前処理した画像をモデルに入力
outputs = model(img)

#print("outputs_keys : ", list(outputs.keys()))                          # ['pred_logits', 'pred_boxes']
#print("outputs['pred_logits'].shape : ", outputs['pred_logits'].shape)  # torch.Size([1, 100, 92])
#print("outputs['pred_boxes'].shape : ", outputs['pred_boxes'].shape)    # torch.Size([1, 100, 4])

# softmax
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]    # 0番目のbatchデータに対して処理を行う（batch_size=1としているから問題ない．）
#print("probas.shape : ", probas.shape)                    # torch.Size([100, 91])


keep = probas.max(-1).values > 0.7
#print("keep : ",keep)                                       # keep : tensor([False, False, False, False,  True, False, False, False, ...
#print("over 0.7 score data num : ", torch.sum(keep).item()) # 17



''' probasはそのまま保持しておき, 閾値を超えるindexと超えないindexをリンストとして獲得するコードを作っちゃった(あんまり必要ないかもkeepでTrue or Falseで保存されている)
selected_indices_list, not_selected_indices_list = utils.get_selected_and_not_selected_indices(probas, threshold=0.7)

#print(f"over 0.7 score data num: {np.shape(selected_indices_list)}, selected_indices_list : {selected_indices_list}")                 # over 0.7 score data num: (17,), selected_indices_list : [4, 18, 31, 34, 35, 44, 47, 49, 64,...
#print(f"under 0.7 score data num: {np.shape(not_selected_indices_list)}, selected_indices_list : {not_selected_indices_list}")        # under 0.7 score data num: (83,), selected_indices_list : [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11,...
'''

bboxes_scaled = utils.rescale_bboxes(outputs['pred_boxes'][0, keep], image.size) # ここでデータが17に集約されている
print("bboxes_scaled.shape : ", bboxes_scaled.shape) #torch.Size([17, 4])

utils.plot_results(image, probas[keep], bboxes_scaled, f"./result/sample_image_{img_idx}", format='png')
