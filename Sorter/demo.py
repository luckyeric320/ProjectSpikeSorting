# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:08:17 2022

@author: liuxj
"""

import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.sorters as ss
import probeinterface as pi
import numpy as np
import os
from ddlraw import Raw
from ddldetect import Detect
from ddlsorted import Sorted
from ddlneurons import Neurons
from mea_256 import dic

recording_name = '20160415_patch2'
Dir = os.path.join('C:/Users/liuxj/Desktop/BMEMajor/zhuanyeshijianzonghexunlian2/raw_data',recording_name)
file_name = 'patch_2_MEA.raw'
offset=1871
data = np.memmap(os.path.join(Dir,file_name),dtype='uint16',offset=offset,mode='r')
data = data.reshape(len(data)//256,256)
data = data[1:60000,0:64]
data = data.astype('float32')
time_series = np.array(data)
Fs = 20000
raw = Raw(time_series,Fs)
raw.scale(1,-2**15-1)
raw.scale(0.1042,0)
raw.set_unit('uV')
locs = [(dic[i]+[0]) for i in range(raw.n_channels)]
raw.set_channels_locs(np.array(locs))
raw = raw.get_spikes()

neurons = raw.sort_by_Klusta()
neurons.plot_neurons_locs()
neurons.plot_neurons_spikes()