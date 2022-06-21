import zipfile
import numpy as np
import os
#读取数据
recording_name = '20160415_patch2'
Dir = os.path.join('C:/Users/liuxj/Desktop/BMEMajor/zhuanyeshijianzonghexunlian2/raw_data',recording_name)
file_name = 'patch_2_MEA.raw'
offset=1871
data = np.memmap(os.path.join(Dir,file_name),dtype='uint16',offset=offset,mode='r')
data = data.reshape(len(data)//256,256)
#选取部分个电极、前10s数据
channels = [17,20,44,51,54,80,81,87,90,114,121,129,222,253,24,47,55,83,86,91,110,111,117,118,192,195,223,225,12,19,25,43,46,52,78,82,88,108,113,116,122,157,193,194,224,16,42,49,53,79,85,92,109,112,123,156,158,197,227,10,18,22]
data = data[1:200000,channels]
data = data.astype('float32')
time_series = np.array(data)
np.save('raw_data.npy',time_series)
Fs = 20000
raw = Raw(time_series,Fs)
raw.scale(1,-2**15-1)
raw.scale(0.1042,0)
raw.set_unit('uV')
#设置电极位置，电极位置信息来自mea_256
with open('mea_256.prb','r',errors='ignore') as f:
    txt = f.read()
txt = txt.replace('channel_groups[1]["channels"]','#channel_groups[1]["channels"]')
exec(txt)
dic = channel_groups[1]["geometry"]
locs = [(dic[i]+[0]) for i in channels]
raw.set_channels_locs(np.array(locs))
np.save('channels_locs.npy',np.array(locs))
with open('info.txt',"w") as info:
    info.write('Fs = 20000\ndrift = -2**15\ngain = 0.1042\nunit = \'uV\'')
with zipfile.ZipFile('upload.zip', 'w') as myzip:      
    myzip.write('info.txt')  
    myzip.write('raw_data.npy')
    myzip.write('channels_locs.npy')
    myzip.close()