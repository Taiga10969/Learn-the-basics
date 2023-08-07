import math
import torch.nn as nn
import matplotlib.pyplot as plt

def cosine_warmup_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr, warmup_epochs):
    if epoch < warmup_epochs:
        lr = min_lr + (init_lr - min_lr) * (epoch / warmup_epochs)
    else:
        lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epoch - warmup_epochs))) + min_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# シンプルな線形モデルを定義する
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # パラメータとして1つの重みと1つのバイアスを持つ線形層を定義します
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # 入力を線形層に渡して2倍にします
        return 2 * self.linear(x)


def plot_learning_rate(optimizer, lr_scheduler, num_epochs):
    # 学習率を記録するためのリスト
    lr_values = []

    # モデルのトレーニングループの中で
    for epoch in range(num_epochs):
        # 学習率を取得
        lr = optimizer.param_groups[0]['lr']
        # 学習率を記録
        lr_values.append(lr)

        # 以下、トレーニングステップなどのコード

        # 学習率の更新
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # 学習率の曲線をプロット
    plt.plot(lr_values)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig('learning_rate_scheduler.svg')
    plt.savefig('learning_rate_scheduler.png')
    plt.close()
