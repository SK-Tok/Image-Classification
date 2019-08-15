#############################################
###### This script is made by SK-Tok ########
#############################################

# Using by Jupyter
% matplotlib inline
import matplotlib.pyplot as plt

# import torch module
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models import ResNet50
#from models import ResNet101
#from models import ResNet152
#from models import VGG16

# import python module
import numpy as np
import random
import argparse
import os
import pdb
from tqdm import tqdm

def make_dataloader(dir_path, batchsize, patchsize, val = False):
    # normalization and data augumentation setting
    if val:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
        
    # make dataloader
    dataset = ImageFolder(dir_path, transform = transform)
    dataloader = DataLoader(
        dataset,
        batchsize = batchsize,
        shuffle = not val,
        num_workers = 4
    )
    
    return dataloader

args = {
    'exp_name':'writing experiment name',
    'data_dir':'wriitng dataset directory',
    'nClass':'writing class number',
    'nEpochs':'writing epoch number',
    'patchsize':'writing patchsize',
    'lr':'writing learning rate',
    'wd':'writing weight decay'
}
print(args)
device = torch.device('cuda')

print('===Loading Datasets===')
train_dataloader = make_dataloader(os.path.join(args['data_dir'], 'train'), args['batchsize'], args['patchsize'])
val_dataloader = make_dataloader(os.path.join(args['data_dir'], 'val'), args['batchsize'], args['patchsize'], val=True)

print('===Building Model===')
model = ResNet50(args['patchsize'], args['nClass']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['wd'])

print('======Networks Initialized======')
print(model)
print('================================')

# train
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    
    for iteration, batch in tqdm(enumerate(train_dataloader, 1)):
        # forward
        input, target = batch
        input, target = input.to(device), taregt.to(device)
        prediction = model(input)
        p = np.argmax(prediction.cpu().data.numpy(),axis=1)
        correct += (p==target).sum().item()
        loss = criterion(prediction, target)
        train_loss += loss.item()
        
        # Update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss = train_loss / len(train_dataloader)
    print('\nEpoch[',epoch,']:train_loss:',train_loss)
    train_acc = correct / len(train_dataloader.dataset)
    print('train_accuracy : {0:.4f}'.format(train_acc))
    return train_acc, train_loss

# validation
def val():
    model.eval()
    val_loss = 0
    correct = 0
    
    for batch in val_dataloader:
        with torch.no_grad():
            input, target = batch
            input, target = input.to(device), target.to(device)
            prediction = model(input)
            p = np.argmax(predction.cpu().data.numpy(), axis=1)
            true_false = (p==target)
            correct += (p==target).sum().item()
            val_loss += criterion(prediction, target)
            
    val_loss = val_loss / len(val_dataloader)
    val_acc = correct / len(val_dataloader.dataset)
    print('test_loss:{:.4f}\n'.format(val_loss))
    print('test_accuracy:{0:.4f}'.format(val_acc))
    
    return val_acc, val_loss

# saved_models
def saved_model(epoch, loss):
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')
        
    path = 'saved_models / {}.csv'.format(args['exp_name'])
    torch.save(model, path)
    print('Best model saved to {}\n'.format(path))
    
    with open(os.path.join('saved_models', 'log'), 'a') as f:
        f.write('{}[Epoch:{:>3}] loss:{:.4f}\n'.format(args['exp_name'],epoch,loss))

min_loss = 100
Train_acc = np.zeros(args['nEpochs'])
Train_loss = np.zeros(args['nEpochs'])

for epoch in range(1, args['nEpochs']+1):
    print(epoch)
    Train_acc([epoch-1], Train_loss[eopch-1]) = train(epoch)
    Val_acc[epoch-1], Val_loss[epoch-1] = val()
    
    if Train_loss[epoch-1] < min_loss:
        saved_model(epoch, Train_loss[epoch-1])
        min_loss = Train_loss[epoch-1]
    
# plot
plt.title('Accuarcy')
plt.plot(Train_acc, label="train")
plt.plot(Val_acc, label="val")
plt.legend(loc = 'upper left')
plt.show()
plt.title('Loss')
plt.plot(Train_loss, label="train")
plt.plot(Val_, label="val")
plt.legend(loc = 'upper left')
plt.show()
