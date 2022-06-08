from ddlraw import Raw
from ddlsorters import *
from zipfile import ZipFile
import numpy as np
#读取zip文件到Raw
file = ZipFile('upload.zip','r')
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
Neurons_list = [[] for i in range(4)]
sorternames =  ['Klusta','Tridesclous','Spykingcircus','Herdingspikes']
for i in range(4):
    sortername = sorternames[i]
    Neurons_list[i] = raw.sort_by(sortername)
    #可视化
    Neurons_list[i].plot_neurons_locs('all',sortername)#二维电极及神经元位置图
#比较分类结果
for i in range(4):
    for j in range(i+1,4):
        gamma_matrix = compare_sorter_results(Neurons_list[i],Neurons_list[j]) 
        for ii in range(Neurons_list[i].n_neurons):
            for jj in range(Neurons_list[j].n_neurons):
                if gamma_matrix[ii][jj] > 0.8:
                    add_str = 'nearly identical with '+sorternames[j]+' neuron '+str(jj)+'\n'
                    Neurons_list[i].neurons_info[ii] += add_str
                    add_str = 'nearly identical with '+sorternames[i]+' neuron '+str(ii)+'\n'
                    Neurons_list[j].neurons_info[jj] += add_str
for i in range(4):
    Neurons_list[i].plot_neurons_spikes('all')#每个神经元在每个电极上的波形图（多图）