from ddlraw import Raw
from ddlsorters import *
from zipfile import ZipFile
import numpy as np
#读取zip文件到Raw
file = ZipFile('./upload.zip','r')
file.extractall()
rd = np.load('raw_data.npy')
cl = np.load('channels_locs.npy')
with open("info.txt", "r") as f:    
    code = f.read() 
exec(code)
raw = Raw(rd,Fs)
raw.scale(1,drift)
raw.scale(gain,0)
raw.set_unit(unit)
if np.shape(cl)[1] == 3:
    raw.set_channels_locs(cl)
elif np.shape(cl)[1] == 2:
    zs = np.zeros(np.shape(cl)[0]).reshape(np.shape(cl)[0],1)
    cl = np.append(cl,zs)
    raw.set_channels_locs(cl)
#滤波
raw = raw.get_spikes()
#使用各种方法分类
Neurons_list = [[] for i in range(3)]
sorternames =  ['Klusta','Tridesclous','Spykingcircus']
gamma = [[] for i in range(3)]
for i in range(3):
    sortername = sorternames[i]
    Neurons_list[i] = raw.sort_by(sortername)
    gamma[i] = np.zeros([Neurons_list[i].n_neurons,1])
#比较分类结果
for i in range(3):
    for j in range(i+1,3):
        gamma_matrix = compare_sorter_results(Neurons_list[i],Neurons_list[j],sorternames[i],sorternames[j]) 
        gamma[i] = np.append(gamma[i],np.max(gamma_matrix,axis=1).reshape(gamma_matrix.shape[0],1),axis=1)
        gamma[j] = np.append(gamma[j],np.max(gamma_matrix,axis=0).reshape(gamma_matrix.shape[1],1),axis=1)
for i in range(3):
    gamma[i] = np.max(gamma[i],axis=1)
#可视化
for i in range(3):
    Neurons_list[i].plot_neurons_locs('all',sortername,sort_by_gamma=True,gamma=gamma[i])#二维电极及神经元位置图
    Neurons_list[i].plot_neurons_spikes('gamma',gamma=gamma[i])#每个神经元在每个电极上的波形图（多图）