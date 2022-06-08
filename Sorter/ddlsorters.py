# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:48:52 2022

@author: liuxj
"""

import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.sorters as ss
import probeinterface as pi
import numpy as np

def Sorter(raw,sorter_name):
    Fs = raw.Fs
    recording = se.NumpyRecording(traces_list = [raw.data],sampling_frequency=Fs,channel_ids = raw.channels_names)
    probe = pi.Probe(ndim=2)
    probe.set_contacts([raw.channels_locs[i,0:2] for i in range(len(raw.channels_locs))])
    probe.set_device_channel_indices(np.arange(raw.n_channels))
    recording = recording.set_probe(probe)
    recording = recording.save(name='recording')
    if sorter_name == 'Klusta':
        sorting = ss.run_klusta(recording, output_folder='tmp_KL')
    elif sorter_name == 'Tridesclous':
        sorting = ss.run_tridesclous(recording, output_folder='tmp_Tr')
    elif sorter_name == 'Spykingcircus':
        sorting = ss.run_spykingcircus(recording,output_folder='tmp_Spy')
    elif sorter_name == 'Herdingspikes':
        sorting = ss.run_herdingspikes(recording,output_folder='tmp_Hd')
    time_points = []
    time_points_s = []
    neurons_result = []
    channel_id = []
    for u in range(len(sorting.unit_ids)):
        neuron_result = []
        time_points += [[]]
        channel_id_tmp = []
        time_points[-1] = sorting.get_unit_spike_train(sorting.unit_ids[u])
        for ch in range(raw.n_channels):
            data = []
            for ii in range(len(time_points[-1])):
                if time_points[-1][ii]+int(0.001*Fs)<=len(raw.data):
                    data += [raw.data[time_points[-1][ii]-int(0.001*Fs):time_points[-1][ii]+int(0.001*Fs),ch]]
                else:
                    time_points[-1] = np.delete(time_points[-1],[ii])
                    break
            result = np.array(data)
            if abs(np.mean(result[:,int(0.001*Fs)]))>np.median(np.abs(raw.data[:,ch]))/0.6745*3:
                neuron_result += [result]
                channel_id_tmp += [raw.channels_names[ch]]
        if neuron_result != []:
            neurons_result += [np.array(neuron_result)]
            channel_id += [channel_id_tmp]
            time_points_s += [time_points[-1]/raw.Fs]
    unit_id = [sorter_name+' '+str(i) for i in range(len(time_points_s))]
    from ddlneurons import Neurons
    neurons = Neurons(neurons_result,time_points_s,unit_id,channel_id,Fs,raw.time_length,raw.channels_names,raw.channels_locs,raw.unit)
    return neurons

def compare_sorter_results(N1,N2):
    gamma = np.zeros([N1.n_neurons,N2.n_neurons])
    for ne1 in range(N1.n_neurons):
        for ne2 in range(N2.n_neurons):
            if len([x for x in N1.channel_id[ne1] if x in N2.channel_id[ne2]])>0:
                gamma[ne1][ne2] = get_gamma(N1.time_points[ne1],N2.time_points[ne2],0.0005) 
    return gamma

def get_gamma(trace1,trace2,tolerance):
    coin = 0
    p = 0
    for tp1 in trace1:
        for p2 in range(p,len(trace2)):
            tp2 = trace2[p2]
            if tp2>tp1-tolerance/2 and tp2<=tp1+tolerance/2:
                coin += 1
            elif tp2>tp1+tolerance/2:
                p = p2
                break
    gamma = (2/(1-2*tolerance*np.max(trace1)/len(trace1)))*((coin-2*tolerance*np.max(trace1)/len(trace1)*len(trace2))/(len(trace1)+len(trace2)))
    return gamma