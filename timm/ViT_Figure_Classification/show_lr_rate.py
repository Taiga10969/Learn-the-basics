import configs
import utils
import dataset

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim


config = configs.Config()

model = utils.SimpleModel()

transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor()])

train_dataset = dataset.ms_FigureClassification(config.dataset_path, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)


optimizer = optim.AdamW(params = model.parameters(),
                        lr=config.lr, 
                        betas=config.betas, 
                        eps=config.eps, 
                        weight_decay=config.weight_decay)

lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                              T_0=config.t_0, 
                                                              T_mult=config.t_mult)

utils.plot_learning_rate(optimizer, lr_scheduler, config.epoch)
