# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:36:56 2022

@author: liuxj
"""
import numpy as np
from scipy import signal

class Detect:
#本类存储峰值检测之后的数据，其中Detect.waveforms为list（len=n_channels），
#list内的每个元素为np.array（shape=[n_spikes,n_samples]），存储每个电极上
#提取到的spike的波形；Detect.time_points为list（len=n_channels），list内
#每个元素为np.array（shape=[n_spikes,]），存储每个spike的时间点（单位:s）
    
    def __init__(self,waveforms,time_points):
        self.waveforms = waveforms
        self.time_points = time_points
        self.n_channels = len(waveforms)
        self.n_spikes = np.zeros([self.n_channels])
        for ch in range(self.n_channels):
            self.n_spikes[ch] = np.shape(waveforms[ch])[0]
        
    def extract_by_median(Raw,k):
        data = Raw.data
        absolute = np.abs(data)
        Fs = Raw.Fs
        med = np.median(absolute,axis=0)
        thres = med/0.6745*k
        time_points=[]
        waveforms=[]
        for i in range(Raw.n_channels):
            samp_points,_ = signal.find_peaks(absolute[:,i],height=thres[i],distance=Fs*0.002)
            for samp_point in samp_points:
                waveform = data[int(np.floor(samp_point-Fs/1000)):int(np.ceil(samp_point+Fs/1000)),i]
                if samp_point == samp_points[0]:
                    waves = np.array([waveform])
                else:
                    waves = np.append(waves,[waveform],axis=0)
            time_points += [samp_points/Fs]
            waveforms += [waves]
        return Detect(waveforms,time_points)
        