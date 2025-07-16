import xlrd
import torch
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import numpy as np

ecl = xlrd.open_workbook('12.xls')
shen = ecl.sheets()
for k in range(3):
    cname = shen[0].cell_value(k+8, 0)
    #cname = cname[0:len(cname)-4]
    print('cname:',cname)
    cdata = torch.zeros(7,7)
    for i in range(7):
        for j in range(7):
            if i*7+j<40:
                cdata[i,j] = float(shen[0].cell_value(k+8, i*7+j+4))
    print('cdata:::',cdata)
    edata = torch.stack([cdata, cdata, cdata], dim=0)
    plt.imsave(cname + '00.png', edata[0])
    ddata = torch.cat([cdata,cdata,cdata,cdata,cdata,cdata,cdata,cdata,cdata], 0)
    ddata = torch.cat([ddata, ddata, ddata, ddata, ddata, ddata, ddata, ddata, ddata], 1)
    ddata = torch.stack([ddata, ddata, ddata], dim=0)
    #ddata = torch.stack([ddata], dim=0)
    print('ddata::::',ddata.shape)
    picdata = np.array(ddata)
    plt.imshow(picdata[0])
    plt.imsave(cname+'.png', picdata[0])

