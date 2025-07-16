import numpy as np
from PIL import Image
import os
import xlrd
import torch
import array
def read_bmp_image(file_path):
    image = Image.open(file_path)
    pixel_array = np.array(image)
    width, height = image.size
    return pixel_array, width, height


ecl = xlrd.open_workbook('12.xls')
shen = ecl.sheets()
for k in range(3):
    cname = shen[0].cell_value(k+8, 0)
    orgdata = []
    #cname = cname[0:len(cname)-4]
    print('cname:',cname)
    # cdata = torch.zeros(7,7)
    # for i in range(7):
    #     for j in range(7):
    #         if i*7+j<40:
    #             cdata[i,j] = float(shen[0].cell_value(k+8, i*7+j+4))
    for j in range(40):
        orgdata.append(shen[0].cell_value(k+8, 4+j))
    print('orgdata::::',orgdata)


    imgdir = './Output/RandomSamples/98/'+cname+'/gen_start_scale=0/'
    for i, filename in enumerate(os.listdir(imgdir)):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            imgname = imgdir+filename
            pix,wid,hei = read_bmp_image(imgname)

            for r in range (9):
                for c in range (9):
                    gendata = []
                    for j in range(6):
                        for l in range(7):
                            gendata.append(pix[j+r*7,l+c*7,0])
                    gendata = gendata[0:40]
                    #print('gendata::::',gendata)
                    diffv = 0
                    for m in range(40):
                        diffv = abs(gendata[m]-orgdata[m])/orgdata[m]+diffv
                    if diffv<16:
                        print('diffv::::',diffv)
            #print(imgname,"::",pix.shape,pix[0:7,0,0])#,pix[0:7],pix[63:70],pix[0:6],pix[0:6],pix[0:6],pix[0:6])

