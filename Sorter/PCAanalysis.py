# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:18:21 2022

@author: liuxj
"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAanalysis:
    
    def __init__(self,Detect,ncomp):
        self.result = PCAanalysis.runpca(Detect,ncomp)
        self.n_channels = Detect.n_channels
        self.n_comp = ncomp
    
    def runpca(Detect,ncomp):
        data = Detect.waveforms
        pca_result = []
        for ch in range(Detect.n_channels):
            ch_data = data[ch]
            pca = PCA(n_components=ncomp)
            pca.fit(ch_data)
            pca_data = pca.transform(ch_data)
            pca_result += [pca_data]
            print('explained_variance_ratio',pca.explained_variance_ratio_)
        return pca_result
    
    def plot_pca(self):
        for ch in range(self.n_channels):
            result = self.result[ch]
            plt.figure()
            if self.n_comp>2:
                ax = plt.axes(projection='3d')
                ax.scatter(result[:, 0], result[:, 1],result[:,2],marker='o')
                plt.show()
            elif self.n_comp==2:
                plt.scatter(result[:, 0], result[:, 1],marker='o')
                plt.show()
            else:
                print('保留的主成分数过少')