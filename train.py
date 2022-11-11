import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import metrics

mean, std = 0.28604059698879553, 0.35302424451492237
batch_size = 256

train_dataset = MNIST('data/MNIST', train=True, download=True,
                                transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = MNIST('data/MNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))


def main():
    epochs = 15
    batch_size = 256
    lr = 1e-3
    weight_decay = 1e-4
    save_path = Path.cwd() / 'save'

    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.cuda()

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size,
                                num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        train_score = train(train_loader, model, optimizer)
        val_score = validate(test_loader, model)

        log_str_train = 'train loss: {:.4f} | train acc: {:.4f}\n'.format(
                  train_score['train_loss'], train_score['train_acc'])
        log_str_val = 'val loss: {:.4f} | val acc: {:.4f}\n'.format(
                  val_score['val_loss'], val_score['val_acc'])

        print(log_str_train)
        print(log_str_val)

        save_obj = {
            'file': __file__,
            'model': model,
            'model_sd': model.state_dict(),
        }

        torch.save(save_obj, save_path / 'epoch{}.pth'.format(epoch))


def train(train_loader, model, optimizer):
    train_loss = metrics.MetricTracker()
    train_acc = metrics.MetricTracker()

    model.train()
    # iterate over data
    for idx, img_data in enumerate(tqdm(train_loader)):
        images = img_data[0].cuda()
        labels = img_data[1].cuda().long()

        logits = model(images)

        loss = F.cross_entropy(logits, labels)
        acc = metrics.compute_acc(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.add(loss.item())
        train_acc.add(acc)

    return {'train_loss': train_loss.get_avg(), 'train_acc': train_acc.get_avg()}


def validate(valid_loader, model):
    val_loss = metrics.MetricTracker()
    val_acc = metrics.MetricTracker()

    model.eval()
    # Iterate over Data
    for img_data in tqdm(valid_loader):
        images = img_data[0].cuda()
        labels = img_data[1].cuda().long()
        with torch.no_grad():
            logits = model(images)

            loss = F.cross_entropy(logits, labels)
            acc = metrics.compute_acc(logits, labels)

        val_loss.add(loss.item())
        val_acc.add(acc)

    return {'val_loss': val_loss.get_avg(), 'val_acc': val_acc.get_avg()}


if __name__ == "__main__":
    main()
