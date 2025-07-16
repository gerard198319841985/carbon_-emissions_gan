import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xlrd
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

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
for k in range(3):
    cname = shen[0].cell_value(k+8, 0)
    #cname = cname[0:len(cname)-4]
    # print('cname:',cname)
    cdata = torch.zeros(7,7)
    for i in range(7):
        for j in range(7):
            if i*7+j<40:
                cdata[i,j] = float(shen[0].cell_value(k+8, i*7+j+4))
    ddata = torch.cat([cdata,cdata,cdata,cdata,cdata,cdata,cdata,cdata,cdata], 0)
    ddata = torch.cat([ddata, ddata, ddata, ddata, ddata, ddata, ddata, ddata, ddata], 1)
    edata = torch.reshape(ddata,(1,1,63,63))
    # print('cdata:::',edata.shape)
    cy = float(shen[0].cell_value(k+8, i+5))
    if k == 0:
        x = edata
    else:
        x = torch.cat((x,edata),dim=0)
    y.append(cy)
y = torch.tensor(y)
# print('x::::::',x.shape)
# print('y:::::', y.shape)

X_train = x #train.data.unsqueeze(1)/255.0
y_train = y #train.targets
# print('X_train::::::',X_train.shape)
# print('y_train::::::',y_train.shape)

trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

X_test = X_train
y_test = y_train



model = LeNet5()
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
lossall = []
for epoch in range(1000):
    for X, y in trainloader:
        pred = model(X)
        y = torch.reshape(y,(3,1))
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
        print(epoch, loss)
        lossall.append(loss)
