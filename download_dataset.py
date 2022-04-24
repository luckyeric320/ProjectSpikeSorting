# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:26:18 2022

@author: liuxj
"""
import numpy as np
import os
from IIR_filter import Raw

recording_name = '20160415_patch2'
Dir = os.path.join('C:/Users/liuxj/Desktop/生医专业/专业实践综合训练2/raw_data',recording_name)
file_name = 'patch_2_MEA.raw'
offset=1871
data = np.memmap(os.path.join(Dir,file_name),dtype='uint16',offset=offset,mode='r')
data = data.reshape(len(data)//256,256)
data = data.astype('float32')

time_series = data
Fs = 20000
print(time_series)
print(Fs)

import matplotlib.pyplot as plt

plt.plot(time_series[0:10000,0])
plt.show()

raw = Raw(time_series,Fs)
spikes = raw.get_spikes()

plt.plot(spikes.data[0:1000,0])
plt.show()
