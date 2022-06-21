# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:18:21 2022

@author: liuxj
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
    
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
    
def plot_pca(pca_results,kmeans_results):
    for ch in range(len(pca_results)):
        result = pca_results[ch]
        color = []
        colorlist = ['r','y','b','g','k']
        for i in range(len(result)):
            color += [colorlist[kmeans_results[ch][i]]]
        plt.figure()
        if result.shape[1]>2:
            ax = plt.axes(projection='3d')
            ax.scatter(result[:, 0], result[:, 1],result[:,2],marker='o',color=color)
            plt.show()
        elif result.shape[1]==2:
            plt.scatter(result[:, 0], result[:, 1],marker='o',color=color)
            plt.show()
        else:
            print('保留的主成分数过少')

def pca_kmeans(D,ncomp,n_cluster):
    pca_result = runpca(D,ncomp)
    kmeans_result = []
    K_means = KMeans(n_clusters=3, init='k-means++')
    for ch in range(D.n_channels):
        K_means.fit(pca_result[ch])
        kmeans_result += [K_means.predict(pca_result[ch])]
    return pca_result, kmeans_result