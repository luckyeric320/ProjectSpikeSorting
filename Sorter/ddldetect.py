# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:36:56 2022

@author: liuxj
"""
import numpy as np
import ddltools.naturalbreaksorter as nbs
import ddltools.PCAanalysis as pcas
import matplotlib.pyplot as plt

class Detect:
#本类存储峰值检测之后的数据，其中Detect.waveforms为list（len=n_channels），
#list内的每个元素为np.array（shape=[n_spikes,n_samples]），存储每个电极上
#提取到的spike的波形；Detect.time_points为list（len=n_channels），list内
#每个元素为np.array（shape=[n_spikes,]），存储每个spike的时间点（单位:s）
    
    def __init__(self,waveforms,time_points,Fs,time_length,unit,channels_names,channels_locs):
        self.waveforms = waveforms
        self.time_points = time_points
        self.Fs = Fs
        self.n_channels = len(waveforms)
        self.time_length = time_length
        self.unit = unit
        self.channels_names = channels_names
        self.channels_locs = channels_locs
        self.n_spikes = np.zeros([self.n_channels]).astype(int)
        for ch in range(self.n_channels):
            self.n_spikes[ch] = np.shape(waveforms[ch])[0]

    def mannual_select_by_peaks(self,ch,lowest,highest):
        waveforms = self.waveforms
        time_points = self.time_points
        for c in ch:
            new_waveforms=np.array([np.zeros_like(waveforms[c][0,:])])
            new_timepoints=[]
            for s in range(int(self.n_spikes[c])):
                peak = waveforms[c][s,int(np.floor(np.shape(waveforms[c])[1]/2))]
                if peak>=lowest[c] and peak<=highest[c]:
                    new_waveforms=np.append(new_waveforms,[waveforms[c][s,:]],axis=0)
                    new_timepoints.append(time_points[c][s])
            new_waveforms = new_waveforms[1:,:]
            waveforms[c] = new_waveforms
            time_points[c] = new_timepoints
        new_detect = Detect(waveforms,time_points,self.Fs,self.time_length,self.unit,self.channels_names,self.channels_locs)
        return new_detect
    
    def sort_by_natural_break(self):
        sortresult = nbs.Natural_breaks(self)
        return sortresult
    
    def sort_by_pca(self,ncomps,n_clusters):
        pca_results, kmeans_results = pcas.pca_kmeans(self,ncomps,n_clusters)
        pcas.plot_pca(pca_results,kmeans_results)
        sorting_result = []
        time_points = []
        neuron_id = []
        for ch in range(self.n_channels):
            sorting_result += [[[] for i in range(n_clusters)]]
            time_points += [[[] for i in range(n_clusters)]]
            sorting_resultl = [[] for i in range(n_clusters)]
            time_pointsl = [[] for i in range(n_clusters)]
            for sp in range(self.n_spikes[ch]):
                sorting_resultl[kmeans_results[ch][sp]] += [self.waveforms[ch][sp]]
                time_pointsl[kmeans_results[ch][sp]] += [self.time_points[ch][sp]]
            for i in range(n_clusters):
                sorting_result[ch][i] = np.array(sorting_resultl[i])
                time_points[ch][i] = np.array(time_pointsl[i])
            neuron_id += [[(str(self.channels_names[ch])+'_pca_sorted'+str(i)) for i in range(n_clusters)]]
        from ddlsorted import Sorted
        S = Sorted(sorting_result,time_points,self.Fs,self.time_length,self.channels_names,self.channels_locs,neuron_id,self.unit)
        return S