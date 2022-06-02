# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:58:58 2022

@author: liuxj
"""
import numpy as np
import matplotlib.pyplot as plt
from ddlneurons import Neurons

class Sorted:
    
    def __init__(self,sorting_result,time_points,Fs,time_length,channels_names,channels_locs,neuron_id):
        self.result = sorting_result
        if type(self.result) is not list:
            raise ValueError('分类结果必须为list of list of numpy.array')
        for i in range(len(self.result)):
            if type(self.result[i]) is not list:
                raise ValueError('分类结果必须为list of list of numpy.array')
            for ii in range(len(self.result[i])):
                if type(self.result[i][ii]) is not np.ndarray:
                    raise ValueError('分类结果必须为list of list of numpy.array')
                if self.result[i][ii].ndim != 2:
                    raise ValueError('分类结果必须为list of list of numpy.array')
        self.time_points = time_points
        if type(self.time_points) is not list:
            raise ValueError('时间点信息必须为list of list of numpy.array')
        if len(self.time_points) != len(self.result):
            raise ValueError('时间点信息长度必须与分类结果匹配')
        for i in range(len(self.time_points)):
            if type(self.time_points[i]) is not list:
                raise ValueError('时间点信息必须为list of list of numpy.array')
            if len(self.time_points[i]) != len(self.result[i]):
                raise ValueError('时间点信息长度必须与分类结果匹配')
            for ii in range(len(self.time_points[i])):
                if type(self.time_points[i][ii]) is not np.ndarray:
                    raise ValueError('时间点信息必须为list of list of numpy.array')
                if self.time_points[i][ii].ndim != 1:
                    raise ValueError('时间点信息必须为list of list of numpy.array')
                if np.shape(self.time_points[i][ii])[0] != np.shape(self.result[i][ii])[0]:
                    raise ValueError('时间点信息长度必须与分类结果匹配')
        self.n_channels = len(sorting_result)
        self.n_neurons = np.zeros([self.n_channels]).astype(int)
        for ch in range(self.n_channels):
            self.n_neurons[ch] = len(sorting_result[ch])
        self.neuron_id = neuron_id
        if type(self.neuron_id) is not list:
            raise ValueError('神经元编号信息必须为列表')
        if len(self.neuron_id) != self.n_channels:
            raise ValueError('神经元编号列表长度必须与通道数一致')
        for i in range(self.n_channels):
            if len(self.neuron_id[i]) != self.n_neurons[i]:
                raise ValueError('神经元编号信息必须与每个通道上的神经元数目一致')
        self.Fs = Fs
        if (type(self.Fs) is not int) and (type(self.Fs) is not float):
            raise ValueError('采样率必须为整数或浮点数')
        self.time_length = time_length
        if (type(self.time_length) is not int) and (type(self.time_length) is not float):
            raise ValueError('记录时间长度必须为整数或浮点数')
        self.channels_names = channels_names
        if type(self.channels_names) is not list:
            raise ValueError('通道名称必须为列表')
        if len(self.channels_names) != self.n_channels:
            raise ValueError('通道名称列表长度必须与通道数一致')
        self.channels_locs = channels_locs
        if (type(self.channels_locs) is not np.ndarray) or (np.shape(self.channels_locs) != (self.n_channels,3)):
            raise ValueError('通道坐标必须为通道数*3的numpy.array')
        self.firing_rate = [np.zeros([self.n_neurons[i]]) for i in range(self.n_channels)]
        for ch in range(self.n_channels):
            for n in range(self.n_neurons[ch]):
                self.firing_rate[ch][n] = len(time_points[ch][n])/time_length
    
    def plot_neuron(self,ch):
        for i in range(self.n_neurons[ch]):
            plt.figure()
            for ii in range(np.shape(self.result[ch][i])[0]):
                plt.plot(self.result[ch][i][ii],color='grey')
            plt.plot(np.mean(self.result[ch][i],axis=0),color='red')
            plt.show()
            
    def to_Neurons(self):
        nidset = []
        result = []
        results = []
        ch_id = []
        time_points = []
        for ch in range(self.n_channels):
            for n in range(self.n_neurons[ch]):
                nid = self.neuron_id[ch][n]
                if (nid in nidset)==False:
                    nidset += [nid]
                    result += [[]]
                    ch_id += [[]]
                    time_points += [[]]
                result[nidset.index(nid)] += [self.result[ch][n]]
                ch_id[nidset.index(nid)] += [self.channels_names[ch]]
                time_points[nidset.index(nid)] = self.time_points[ch][n]
        for n in range(len(result)):
            results += [[]]
            results[n] = np.array(result[n])
        return Neurons(results,time_points,nidset,ch_id,self.Fs,self.time_length,self.channels_names,self.channels_locs)
    
