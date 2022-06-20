# 读取规定的上传格式的文件并完整运行流程

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
    #以下内容绘制到‘./str(sorternames[i])’
    for t in range(int(Neurons_list[i].time_length)):
        Neurons_list[i].plot_neurons_locs([t,t+1],sortername,sort_by_gamma=True,gamma=gamma[i])
    Neurons_list[i].plot_neurons_locs('all',sortername,sort_by_gamma=True,gamma=gamma[i])
    #以下内容绘制到‘./str(sorternames[i]/wave)’
    Neurons_list[i].plot_neurons_spikes('gamma',gamma=gamma[i])#每个神经元在每个电极上的波形图（多图）

#生成下载文件
import os
for i in range(3):
    os.makedirs('output_result/'+sorternames[i],exist_ok=True)
    for ne in range(Neurons_list[i].n_neurons):
        addstr = str(Neurons_list[i].neuron_id[ne])
        np.save(os.path.join('output_result/'+sorternames[i],'waveforms_'+addstr+'.npy'),Neurons_list[i].result[ne])
        np.save(os.path.join('output_result/'+sorternames[i],'timeseries_'+addstr+'.npy'),Neurons_list[i].time_points[ne])
    np.save(os.path.join('output_result/'+sorternames[i],'neurons_locs.npy'),Neurons_list[i].neurons_locs)

import zipfile 
import sys 

def writeAllFileToZip(absDir,zipFile):
    for f in os.listdir(absDir):
        absFile=os.path.join(absDir,f) 
        if os.path.isdir(absFile): 
            relFile=absFile[len(os.getcwd())+1:] 
            zipFile.write(relFile) 
            writeAllFileToZip(absFile,zipFile) 
        else: 
            relFile=absFile[len(os.getcwd())+1:] 
            zipFile.write(relFile)
    return
    
zipFilePath=os.path.join(sys.path[0],"output_result.zip") 
zipFile=zipfile.ZipFile(zipFilePath,"w",zipfile.ZIP_DEFLATED) 
absDir=os.path.join(sys.path[0],"output_result") 
writeAllFileToZip(absDir,zipFile) 
print("压缩成功")