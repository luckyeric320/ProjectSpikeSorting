# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:51:06 2022

@author: liuxj
"""

import numpy as np
import jenkspy
from scipy import stats,signal
import matplotlib.pyplot as plt

def burst_finding(N,R,thr):
    tp = N.time_points.copy()
    n_result = [[] for i in range(N.n_neurons)]
    for ne in range(N.n_neurons):
        time_points = N.time_points[ne] 
        if len(N.channel_id[ne])>1:
            result = []
            for ch in range(len(N.channel_id[ne])):
                result += [np.abs(np.mean(N.result[ne][ch][:,int(np.floor(np.shape(N.result[ne][ch])[1]/2))]))]
            qresult = sorted(result,reverse=True)
            channel_remain = N.channel_id[ne][result.index(qresult[0])]
            result_remain = N.result[ne][result.index(qresult[0])]
        else:
            channel_remain = N.channel_id[ne]
            result_remain = N.result[ne]
        result_remain = np.array(result_remain)
        noise_trace = []
        for sp in range(1,len(N.time_points[ne])):
            if N.time_points[ne][sp]-N.time_points[ne][sp-1]>0.004:
                noise_trace += [[]]
                noise_trace[-1] = np.array([])

                for ch in channel_remain:
                    noise_trace[-1] = np.append(noise_trace[-1],R.data[int(np.floor((N.time_points[ne][sp]-0.003)*R.Fs)):int(np.floor((N.time_points[ne][sp]-0.003)*R.Fs)+0.002*R.Fs),R.channels_names.index(ch)])
        noise_trace = np.array(noise_trace)
        C = np.zeros([np.shape(noise_trace)[1],np.shape(noise_trace)[1]])
        for i in range(np.shape(noise_trace)[1]):
            for j in range(np.shape(noise_trace)[1]):
                C[i,j] = np.cov(noise_trace[:,i],noise_trace[:,j])[0][1]
        Ci = np.linalg.inv(C)
        isi = sorted(np.diff(N.time_points[ne]))
        for nbc in range(2,len(isi)):
            isith = jenkspy.jenks_breaks(isi, nb_class=nbc)[1]
            if isith < 0.1:
                break
        #怎样寻找断点更好？
        print('isi_threshold=',isith)
        amp = np.array([])
        Y = np.array([])
        isi_mean = 0
        n_of_spike = 0
        for sp in range(1,len(N.time_points[ne])):
            if N.time_points[ne][sp]-N.time_points[ne][sp-1]<isith:
                isi_mean += N.time_points[ne][sp]-N.time_points[ne][sp-1]
                n_of_spike += 1
            else:
                if n_of_spike>0:
                    isi_mean /= n_of_spike
                    for nn in range(n_of_spike-1):
                        amp = np.append(amp,result_remain[:,sp-nn-1,int(np.floor(np.shape(result_remain)[2]/2))]/np.mean(result_remain[:,:,int(np.floor(np.shape(result_remain)[2]/2))],axis=1))
                        for nch in range(len(result_remain[:,sp-nn-1,int(np.floor(np.shape(result_remain)[2]/2))]/np.mean(result_remain[:,:,int(np.floor(np.shape(result_remain)[2]/2))],axis=1))):
                            Y = np.append(Y,isi_mean/(n_of_spike-nn)/isith)
                isi_mean = 0
                n_of_spike = 0
        if np.shape(amp)[0]>0:
            slope,_,_,_,_ = stats.linregress(np.log(Y),np.log(amp))
            print('lamda=',slope)
            if slope>0:
                ksi = np.mean(result_remain[:,:,:],axis=1)
                plt.plot(ksi[0])
                plt.show()
                ksi = ksi.reshape((np.shape(ksi)[0]*np.shape(ksi)[1]))
                ksit = ksi.reshape(-1,1)
                n_of_spike = 1
                for sp in range(1,len(N.time_points[ne])):
                    if N.time_points[ne][sp]-N.time_points[ne][sp-1]<isith:
                        isi_mean += N.time_points[ne][sp]-N.time_points[ne][sp-1]
                        n_of_spike += 1
                    else:
                        if n_of_spike >1:
                            isi_mean /= (n_of_spike-1)
                        else:
                            isi_mean = isith
                        t0 = (N.time_points[ne][sp-1]+0.001)*R.Fs
                        for nnn in range(5):
                            if (t0/R.Fs+0.001)>N.time_points[ne][sp]:
                                break
                            D = np.zeros(int(np.floor(isi_mean*R.Fs)))
                            mu = (isi_mean/(n_of_spike+nnn)/isith)**slope
                            for t in range(int(np.floor(isi_mean*R.Fs))):
                                X = np.array([])
                                for ch in channel_remain:
                                    X = np.append(X,R.data[int(t0+t):int(t0+t)+np.shape(result_remain)[2],R.channels_names.index(ch)])
                                D[t] = mu*(X.dot(np.dot(Ci,ksit))-mu/2*ksi.dot(np.dot(Ci,ksit)))+np.log(0.5)
                            peaks,_ = signal.find_peaks(D,height=thr)
                            #阈值设在多少合适呢？
                            if len(peaks)>0:
                                t0 += (peaks[0]+0.001*R.Fs)
                                #print(peaks[0]/R.Fs)
                                plt.plot(R.data[int(t0-np.shape(result_remain)[2]/2):int(t0-np.shape(result_remain)[2]/2)+np.shape(result_remain)[2]+1,R.channels_names.index(ch)])
                                time_points = np.append(time_points,(t0/R.Fs))
                            else:
                                break                                                            
                        n_of_spike = 1
                        isi_mean = 0
                plt.show()
                time_points.sort()
                tp[ne] = time_points
                for ch in range(len(N.channel_id[ne])):
                    n_result[ne] += [[]]
                    for t in tp[ne]:
                        n_result[ne][ch] += [R.data[int(t*R.Fs-np.shape(result_remain)[2]/2):int(t*R.Fs-np.shape(result_remain)[2]/2)+np.shape(result_remain)[2]+1,R.channels_names.index(N.channel_id[ne][ch])]]
    return tp,n_result
