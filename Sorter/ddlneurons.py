# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:43:36 2022

@author: liuxj
"""
import numpy as np
import burst_finding

class Neurons:
    
    def __init__(self,result,time_points,nidset,ch_id,Fs,time_length,channels_names,channels_locs):
        self.result = result
        self.time_points = time_points
        self.neuron_id = nidset
        self.channel_id = ch_id
        self.Fs = Fs
        self.time_length = time_length
        self.channels_names = channels_names
        self.channels_locs = channels_locs
        self.n_channels = len(channels_names)
        self.n_neurons = len(nidset)
        self.firing_rate = np.zeros([self.n_neurons])
        for n in range(self.n_neurons):
                self.firing_rate[n] = len(time_points[n])/time_length
                
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
        return Sorted(result,time_points,self.Fs,self.time_length,self.channels_names,self.channels_locs,neuron_id)
                
    def find_bursts(self,R,thr):
        ntp,nresult = burst_finding.burst_finding(self,R,thr)
        neo_neurons = Neurons(nresult,ntp,self.neuron_id,self.channel_id,self.Fs,self.time_length,self.channels_names,self.channels_locs)
        return neo_neurons 
