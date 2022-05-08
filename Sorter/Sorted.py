# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:58:58 2022

@author: liuxj
"""
import numpy as np
import jenkspy
import Detect

class Sorted:
    
    def __init__(self,sorting_result,time_points):
        self.result = sorting_result
        self.time_points = time_points
        self.n_channels = len(sorting_result)
        self.n_neurons = np.zeros([self.n_channels]).astype(int)
        for ch in range(self.n_channels):
            self.n_neurons[ch] = len(sorting_result[ch])
        
    def Natural_breaks(Detect):
        sorted_spikes = [[] for i in range(Detect.n_channels)]
        time_points = [[] for i in range(Detect.n_channels)]
        for ch in range(Detect.n_channels):
            peak = np.zeros([Detect.n_spikes[ch]])
            for s in range(Detect.n_spikes[ch]):
                waveforms = Detect.waveforms[ch]
                peak[s] = waveforms[s,int(np.floor(np.shape(waveforms)[1]/2))]
            nb_class = Sorted.select_n_groups(peak)
            breaks = jenkspy.jenks_breaks(peak, nb_class=nb_class)
            print(breaks)
            sorted_spikes[ch] = [[] for i in range(nb_class)]
            time_points[ch] = [[] for i in range(nb_class)]
            for s in range(Detect.n_spikes[ch]):
                for nb in range(nb_class):
                    if peak[s]<=breaks[nb+1] and peak[s]>breaks[nb] or (nb==0 and peak[s]==breaks[0]):
                        sorted_spikes[ch][nb] += [waveforms[s,:]]
                        time_points[ch][nb] += [Detect.time_points[ch][s]]
        return Sorted(sorted_spikes,time_points)
    
    def plot_neuron(self,ch,n_neuron):
        if n_neuron=='all':
            for i in range(self.n_neurons[ch]):
                Detect.plot_average(np.array(self.result[ch][i]))
            
    def goodness_of_variance_fit(array, classes):
        classes = jenkspy.jenks_breaks(array, classes)
        classified = np.array([Sorted.classify(i, classes) for i in array])
        maxz = max(classified)
        zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
        sdam = np.sum((array - array.mean()) ** 2)
        array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
        sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])
        gvf = (sdam - sdcm) / sdam
        return gvf
    
    def classify(value, breaks):
        for i in range(1, len(breaks)):
            if value < breaks[i]:
                return i
        return len(breaks) - 1
    
    def select_n_groups(array):
        gvf = 0.0
        nclasses = 2
        while gvf < .7:
            gvf = Sorted.goodness_of_variance_fit(array, nclasses)
            nclasses += 1
        print('Divide into '+str(nclasses)+' classes')
        return nclasses