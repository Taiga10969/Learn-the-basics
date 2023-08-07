import configs
import dataset
import utils

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

config = configs.Config()

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')


model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=3)
model = nn.DataParallel(model)
model.to(device)


criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = optim.AdamW(params = model.parameters(),
                        lr=config.lr, 
                        betas=config.betas, 
                        eps=config.eps, 
                        weight_decay=config.weight_decay)

lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                              T_0=config.t_0, 
                                                              T_mult=config.t_mult)

print('Create of models, etc.')
#print("model : ", model)
#print("criterion : ", criterion)
#print("optimizer : ", optimizer)

transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor()])


train_dataset = dataset.ms_FigureClassification(config.dataset_path, train=True, transform=transform)
test_dataset = dataset.ms_FigureClassification(config.dataset_path, train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


best_test_loss = 10.0

for epoch in range(1, config.epoch+1, 1):
    with tqdm(train_loader) as pbar:
        pbar.set_description(f'[train epoch : {epoch}]')
        model.train()
        sum_train_loss = 0.0
        train_loss = 0.0
        train_count = 0

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            train_count += 1
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_train_loss += loss.item()
            train_loss = sum_train_loss / train_count

            pbar.set_postfix(OrderedDict(loss=loss.item(), ave_loss=train_loss, lr = optimizer.param_groups[0]['lr']))


    with tqdm(test_loader) as pbar:
        with torch.no_grad():
            pbar.set_description(f'[test epoch : {epoch}]')
            model.eval()
            sum_test_loss = 0.0
            test_loss = 0.0
            test_count = 0
            correct = 0

            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                test_count += 1
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                sum_test_loss += loss.item()

                pred = torch.argmax(outputs, dim=1)
                correct += torch.sum(pred == labels)

                test_loss = sum_test_loss / test_count
                acc = correct / test_count
                
                pbar.set_postfix(OrderedDict(loss=loss.item(), ave_loss=test_loss, accuracy_rate=acc))
            
    lr_scheduler.step()
    with open(config.record_dir+'/loss_record.csv', 'a') as f:
        print(f'{epoch}, {train_loss}, {test_loss}, {acc}', file=f)         
    
    if best_test_loss > test_loss:
        best_test_loss = test_loss
        torch.save(model.module.state_dict(), config.record_dir+"/train_best.pth")   
            
