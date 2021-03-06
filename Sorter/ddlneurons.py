# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:43:36 2022

@author: liuxj
"""
import numpy as np
from ddltools import burst_finding
import matplotlib.pyplot as plt
import os

class Neurons:
    
    def __init__(self,result,time_points,nidset,ch_id,Fs,time_length,channels_names,channels_locs,unit):
        self.result = result
        if type(self.result) is not list:
            raise ValueError('分类结果必须为list of numpy.array')
        for i in range(len(self.result)):
            if type(self.result[i]) is not np.ndarray:
                raise ValueError('分类结果必须为list of numpy.array')
            if self.result[i].ndim != 3:
                raise ValueError('分类结果必须为list of numpy.array')
        self.time_points = time_points
        if type(self.time_points) is not list:
            raise ValueError('时间点信息必须为list of numpy.array')
        if len(self.time_points) != len(self.result):
            raise ValueError('时间点信息长度必须与分类结果匹配')
        for i in range(len(self.time_points)):
            if type(self.time_points[i]) is not np.ndarray:
                raise ValueError('时间点信息必须为list of numpy.array')
            if self.time_points[i].ndim != 1:
                raise ValueError('时间点信息必须为list of numpy.array')
            if np.shape(self.time_points[i])[0] != np.shape(self.result[i])[1]:
                raise ValueError('时间点信息长度必须与分类结果匹配')
        self.neuron_id = nidset
        if type(self.neuron_id) is not list:
            raise ValueError('神经元编号信息必须为列表')
        if len(self.neuron_id) != len(self.result):
            raise ValueError('神经元编号列表长度必须与神经元数一致')
        self.channel_id = ch_id
        if type(self.channel_id) is not list:
            raise ValueError('通道编号信息必须为list of list')
        if len(self.channel_id) != len(self.result):
            raise ValueError('通道编号信息必须与分类结果维度匹配')
        for i in range(len(self.channel_id)):
            if type(self.channel_id[i]) is not list:
                raise ValueError('通道编号信息必须为list of list')
            if len(self.channel_id[i]) != np.shape(self.result[i])[0]:
                raise ValueError('通道编号信息必须与分类结果维度匹配')
        self.Fs = Fs
        if (type(self.Fs) is not int) and (type(self.Fs) is not float):
            raise ValueError('采样率必须为整数或浮点数')
        self.time_length = time_length
        if (type(self.time_length) is not int) and (type(self.time_length) is not float):
            raise ValueError('记录时间长度必须为整数或浮点数')
        self.channels_names = channels_names
        if type(self.channels_names) is not list:
            raise ValueError('通道名称必须为列表')
        self.n_channels = len(channels_names)
        self.channels_locs = channels_locs
        if (type(self.channels_locs) is not np.ndarray) or (np.shape(self.channels_locs) != (self.n_channels,3)):
            raise ValueError('通道坐标必须为通道数*3的numpy.array')
        self.n_neurons = len(nidset)
        self.firing_rate = np.zeros([self.n_neurons])
        for n in range(self.n_neurons):
            self.firing_rate[n] = len(time_points[n])/time_length
        self.neurons_locs = np.zeros([self.n_neurons,3])
        for ne in range(self.n_neurons):
            a = 0
            for ch in range(len(self.channel_id[ne])):
                self.neurons_locs[ne] += self.channels_locs[self.channels_names.index(self.channel_id[ne][ch])]*np.abs(np.mean(self.result[ne][ch][:,int(np.floor(np.shape(self.result[ne][ch])[1]/2))]))
                a += np.abs(np.mean(self.result[ne][ch][:,int(np.floor(np.shape(self.result[ne][ch])[1]/2))]))
            self.neurons_locs[ne] = self.neurons_locs[ne]/a
        self.unit = unit
        self.neurons_info = ['' for i in range(self.n_neurons)]
            
    def to_Sorted(self):
        from ddlsorted import Sorted
        result = [[] for i in range(len(self.channels_names))]
        time_points = [[] for i in range(len(self.channels_names))]
        neuron_id = [[] for i in range(len(self.channels_names))]
        channels_names = self.channels_names
        for n in range(self.n_neurons):
            for ch in range(len(self.channel_id[n])):
                chn = self.channel_id[n][ch]
                result[channels_names.index(chn)] += [self.result[n][ch]]
                time_points[channels_names.index(chn)] += [self.time_points[n]]
                neuron_id[channels_names.index(chn)] += [self.neuron_id[n]]
        return Sorted(result,time_points,self.Fs,self.time_length,self.channels_names,self.channels_locs,neuron_id,self.unit)
                
    def find_bursts(self,R,thr):
        ntp,nresult = burst_finding.burst_finding(self,R,thr)
        neo_neurons = Neurons(nresult,ntp,self.neuron_id,self.channel_id,self.Fs,self.time_length,self.channels_names,self.channels_locs,self.unit)
        return neo_neurons 

    def get_firing_rate(self,timestep):
        firing_rate_curve = np.zeros([self.n_neurons,int(np.floor(self.time_length/timestep))])
        for ne in range(self.n_neurons):
            ns = 0
            for t in range(np.shape(firing_rate_curve)[1]):
                for n_s in range(ns,len(self.time_points[ne])):
                    if self.time_points[ne][n_s]<=(t+1)*timestep:
                        firing_rate_curve[ne][t] += 1
                    else:
                        ns = n_s
                        break
        firing_rate_curve = firing_rate_curve/timestep  
        return firing_rate_curve
    
    def plot_neurons_locs(self,time,title,sort_by_gamma=False,gamma=None,savefig=False,directory=None):
        plt.scatter(self.channels_locs[:,0],self.channels_locs[:,1],c='y',marker='s')
        if time == 'all':
            fr = self.firing_rate
        elif (len(time) == 2) and ((type(time[0]) is float) or (type(time[0]) is int)) and ((type(time[1]) is float) or (type(time[1]) is int)):
            fr = np.zeros([self.n_neurons])
            for ne in range(self.n_neurons):
                for n_s in range(0,len(self.time_points[ne])):
                    if (self.time_points[ne][n_s]>=time[0]) and (self.time_points[ne][n_s]<time[1]):
                        fr[ne] += 1
                    elif self.time_points[ne][n_s]>=time[1]:
                        break
            fr = fr/(time[1]-time[0])
        if sort_by_gamma == False:
            plt.scatter(self.neurons_locs[:,0],self.neurons_locs[:,1],c=fr,cmap='Reds',marker='*')
        elif sort_by_gamma == True:
            for ne in range(self.n_neurons):
                if gamma[ne] > 0.8:
                    plt.scatter(self.neurons_locs[ne,0],self.neurons_locs[ne,1],c=fr[ne],cmap='Reds',marker='*',vmin=0,vmax=max(fr))
                else:
                    plt.scatter(self.neurons_locs[ne,0],self.neurons_locs[ne,1],c=fr[ne],cmap='Greens',marker='*',vmin=0,vmax=max(fr))
        plt.title(title)
        if savefig == True:
            plt.savefig(os.path.join(directory,'locs_'+str(time)+'.png'))
        plt.show()
        
    def plot_neurons_spikes(self,n_id,gamma=None,savefig=False,directory=None):
        if n_id == 'all':
            nelist = [i for i in range(self.n_neurons)]
        elif n_id == 'gamma':
            nelist = np.argsort(-gamma)
        elif type(n_id) is int:
            nelist = [n_id]
        elif type(n_id) is str:
            nelist = [self.neuron_id.index(n_id)]
        for ne in nelist:
            print('当前神经元ID为'+str(self.neuron_id[ne]))
            print('在'+str(len(self.channel_id[ne]))+"个通道上被记录到")
            plt.figure()
            plt.scatter(self.channels_locs[:,0],self.channels_locs[:,1],c='y',marker='s')
            plt.scatter(self.neurons_locs[ne,0],self.neurons_locs[ne,1],c='r',marker='*')
            plt.title(self.neuron_id[ne]+'\n'+self.neurons_info[ne])
            if savefig == True:
                plt.savefig(os.path.join(directory,'Neuron_'+str(self.neuron_id[ne])+' _loc.png'),bbox_inches='tight')
            for i in range(len(self.channel_id[ne])):
                plt.figure()
                for ii in range(np.shape(self.result[ne][i])[0]):
                    plt.plot(self.result[ne][i][ii],color='grey')
                plt.plot(np.mean(self.result[ne][i],axis=0),color='red')
                plt.xlabel('time(*1/Fs)')
                plt.ylabel(self.unit)
                plt.title('Neuron:'+str(self.neuron_id[ne])+' '+'Channel:'+str(self.channel_id[ne][i]))
                if savefig == True:
                    plt.savefig(os.path.join(directory,'Neuron_'+str(self.neuron_id[ne])+'_'+'Channel_'+str(self.channel_id[ne][i])+'.png'),bbox_inches='tight')
                plt.show()