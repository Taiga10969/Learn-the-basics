import os
import json
from torch.utils.data import Dataset

from PIL import Image
from sklearn.model_selection import train_test_split

class ms_FigureClassification(Dataset):
    def __init__(self, dataset_path, train=True, transform=None):

        self.dataset_path = dataset_path
        self.train = train
        self.transform = transform

        self.annotation_file = os.path.join(dataset_path, 'annotation_files.json')    
        self.data_list = []
        self.labels = []
        
        with open(self.annotation_file) as f:
            annotation = json.load(f)
            for entry in annotation["annotations"]:
                img_path = entry["img_path"]
                category = entry["category"]
                self.data_list.append(img_path)
                self.labels.append(category)
        

        # データをトレーニングデータとテストデータに分割
        train_size = 0.9  # トレーニングデータの割合
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.data_list, self.labels, train_size=train_size, stratify=self.labels)

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, index):

        if self.train == True:
            img_path = self.train_data[index]
            label_id = self.train_labels[index]
        else:
            img_path = self.test_data[index]
            label_id = self.test_labels[index]
        
        img_path = os.path.join(self.dataset_path, img_path)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image, label_id
    


if __name__ == '__main__':
    import configs
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    config = configs.Config()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = ms_FigureClassification(config.dataset_path, train=True, transform=transform)
    test_dataset = ms_FigureClassification(config.dataset_path, train=False, transform=transform)

    # ex)
    image, label_id = train_dataset[0]

    print('len(train_dataset) : ', len(train_dataset))
    print('len(test_dataset) : ', len(test_dataset))
    print('ex) train_dataset[0] : ')
    print('image : ', image)
    print('label_id : ', label_id)

    print('create of dataloader ... ')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    images, labels = next(iter(train_loader))
    print('images : ', images)
    print('labels : ', labels)
