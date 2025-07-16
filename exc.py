import xlrd
import torch
from torchvision import utils as vutils


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    #print('len(input_tensor.shape):',len(input_tensor.shape),'input_tensor.shape[0]:',input_tensor.shape[0])
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


ecl = xlrd.open_workbook('13.xls')
shen = ecl.sheets()
for k in range(3):
    cname = shen[0].cell_value(k+2, 1)
    cname = cname[0:len(cname)-4]
    print('cname:',cname)
    cdata = torch.zeros(13,13)
    for i in range(13):
        for j in range(13):
            cdata[i,j] = float(shen[i].cell_value(k+2, j+3))
    ddata = torch.cat([cdata,cdata,cdata,cdata,cdata,cdata,cdata], 0)
    ddata = torch.cat([ddata, ddata, ddata, ddata, ddata, ddata, ddata], 1)
    ddata = torch.stack([ddata, ddata, ddata], dim=0)
    ddata = torch.stack([ddata], dim=0)
    save_image_tensor(ddata,str(cname)+'.bmp')
