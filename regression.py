import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xlrd
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=40, n_hidden=10, n_output=1)

print(net)

ecl = xlrd.open_workbook('12.xls')
shen = ecl.sheets()
x = np.array([])
y = np.array([])
for k in range(3):
    cname = shen[0].cell_value(k+8, 0)
    print('cname:',cname)
    cdata = torch.empty(40)
    for i in range(40):
        cdata[i] = float(shen[0].cell_value(k+8, i+4))
    print('cdata:::',cdata)
    cy = float(shen[0].cell_value(k+8, i+5))
    print('cy:::::',cy)
    x = np.concatenate((x,cdata))
    y = [y,cy]
x = np.reshape(x,(3,-1))
print('x::::::',x,np.size(x))

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#torch.save(net, 'net.pkl')

