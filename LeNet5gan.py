import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import os
import cv2

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        #self.fc1 = nn.Linear(16*5*5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(len(x), -1)
        #print('x0::::', x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


ecl = xlrd.open_workbook('12.xls')
shen = ecl.sheets()
y = []
x = torch.tensor([])
for k in range(3):
    cname = shen[0].cell_value(k+8, 0)
    imgdir = './Output/RandomSamples/99/' + cname + '/gen_start_scale=0/'
    for i, filename in enumerate(os.listdir(imgdir)):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            imgname = imgdir + filename
        ddata = cv2.imread(imgname)
        ddata = torch.tensor(ddata)
        adata, bdata, cdata = ddata.split(1, dim=2)
        cdata = cdata.reshape([1,1,63,63])
        #print('cdata:::',cdata.shape)
        cy = float(shen[0].cell_value(k+8, 39+5))
        x = torch.cat((x,cdata),dim=0)
        y.append(cy)
y = torch.tensor(y)
print('x::::::',x.shape)
print('y:::::', y.shape)

X_train = x #train.data.unsqueeze(1)/255.0
y_train = y #train.targets
# print('X_train::::::',X_train.shape)
# print('y_train::::::',y_train.shape)

trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=3000, shuffle=True)

X_test = X_train
y_test = y_train



model = LeNet5()
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
lossall = []
for epoch in range(10000):
    for X, y in trainloader:
        pred = model(X)
        y = torch.reshape(y,(3000,1))
        pred = pred.to(torch.float32)
        y = y.to(torch.float32)
        #print('pred.shape:', pred.shape, y.shape)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        y_pred = model(X_train)
        acc_train = (y_pred.argmax(dim=1) == y_train).float().mean().item()
        y_pred = model(X_test)
        acc_test = (y_pred.argmax(dim=1) == y_test).float().mean().item()
        print(epoch, loss/30000)
        lossall.append(loss)
