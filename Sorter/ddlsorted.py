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
        self.time_points = time_points
        self.n_channels = len(sorting_result)
        self.n_neurons = np.zeros([self.n_channels]).astype(int)
        self.neuron_id = neuron_id
        for ch in range(self.n_channels):
            self.n_neurons[ch] = len(sorting_result[ch])
        self.Fs = Fs
        self.time_length = time_length
        self.channels_names = channels_names
        self.channels_locs = channels_locs
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
        return Neurons(result,time_points,nidset,ch_id,self.Fs,self.time_length,self.channels_names,self.channels_locs)
    
