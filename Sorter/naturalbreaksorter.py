# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:28:13 2022

@author: liuxj
"""
import numpy as np
import jenkspy
from ddlsorted import Sorted

def Natural_breaks(D):
        sorted_spikes = [[] for i in range(D.n_channels)]
        sorted_spikel = [[] for i in range(D.n_channels)]
        time_points = [[] for i in range(D.n_channels)]
        neuron_id = [[] for i in range(D.n_channels)]
        for ch in range(D.n_channels):
            peak = np.zeros([D.n_spikes[ch]])
            for s in range(D.n_spikes[ch]):
                waveforms = D.waveforms[ch]
                peak[s] = waveforms[s,int(np.floor(np.shape(waveforms)[1]/2))]
            nb_class = select_n_groups(peak)
            breaks = jenkspy.jenks_breaks(peak, nb_class=nb_class)
            print(breaks)
            sorted_spikel[ch] = [[] for i in range(nb_class)]
            sorted_spikes[ch] = [[] for i in range(nb_class)]
            time_points[ch] = [[] for i in range(nb_class)]
            for s in range(D.n_spikes[ch]):
                for nb in range(nb_class):
                    if peak[s]<=breaks[nb+1] and peak[s]>breaks[nb] or (nb==0 and peak[s]==breaks[0]):
                        sorted_spikel[ch][nb] += [waveforms[s,:]]
                        time_points[ch][nb] += [D.time_points[ch][s]]
            for nb in range(nb_class):
                sorted_spikes[ch][nb] = np.array(sorted_spikel[ch][nb])
            neuron_id[ch] = ['ch'+str(ch)+'_n'+str(nb) for nb in range(nb_class)]
        return Sorted(sorted_spikes,time_points,D.Fs,D.time_length,D.channels_names,D.channels_locs,neuron_id)
    
def goodness_of_variance_fit(array, classes):
        classes = jenkspy.jenks_breaks(array, classes)
        classified = np.array([classify(i, classes) for i in array])
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
    while gvf < .6:
        gvf = goodness_of_variance_fit(array, nclasses)
        nclasses += 1
    print('Divide into '+str(nclasses)+' classes')
    return nclasses
