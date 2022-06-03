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

def KlustaSorter(raw):
    Fs = raw.Fs
    recording = se.NumpyRecording(traces_list = [raw.data],sampling_frequency=Fs,channel_ids = raw.channels_names)
    probe = pi.Probe(ndim=2)
    probe.set_contacts([raw.channels_locs[i,0:2] for i in range(len(raw.channels_locs))])
    probe.set_device_channel_indices(np.arange(raw.n_channels))
    recording = recording.set_probe(probe)
    sorting_KL = ss.run_klusta(recording, output_folder='tmp_KL')
    time_points = []
    time_points_s = []
    neurons_result = []
    channel_id = []
    unit_id = []
    for u in range(len(sorting_KL.unit_ids)):
        neuron_result = []
        time_points += [[]]
        channel_id_tmp = []
        time_points[-1] = sorting_KL.get_unit_spike_train(sorting_KL.unit_ids[u])
        for ch in range(raw.n_channels):
            data = []
            for ii in range(len(time_points[-1])):
                if time_points[-1][ii]+int(0.001*Fs)<=len(raw.data):
                    data += [raw.data[time_points[-1][ii]-int(0.001*Fs):time_points[-1][ii]+int(0.001*Fs),ch]]
                else:
                    time_points[-1] = np.delete(time_points[-1],[ii])
            result = np.array(data)
            if abs(np.mean(result[:,int(0.001*Fs)]))>np.median(np.abs(raw.data[:,ch]))/0.6745*3:
                neuron_result += [result]
                channel_id_tmp += [raw.channels_names[ch]]
        if neuron_result != []:
            neurons_result += [np.array(neuron_result)]
            channel_id += [channel_id_tmp]
            time_points_s += [time_points[-1]/raw.Fs]
            unit_id += [sorting_KL.unit_ids[u]]
    from ddlneurons import Neurons
    neurons = Neurons(neurons_result,time_points_s,unit_id,channel_id,Fs,raw.time_length,raw.channels_names,raw.channels_locs,raw.unit)
    return neurons