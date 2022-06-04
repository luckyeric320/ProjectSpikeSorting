from scipy import signal
import numpy as np
from ddldetect import Detect
import ddlsorters

class Raw:
#本类存储原始波形数据，格式为np.array(dtype='float32')，shape=[n_samples,n_channels]
    
    def __init__(self,time_series,Fs):
        self.data = time_series
        if type(self.data) is not np.ndarray:
            raise ValueError('输入数据必须为numpy.array')
        self.n_channels = np.shape(time_series)[1]
        self.Fs = Fs
        if (type(self.Fs) is not int) and (type(self.Fs) is not float):
            raise ValueError('采样率必须为整数或浮点数')
        self.time_length = np.shape(self.data)[0]/Fs
        self.unit = ''
        self.channels_names = [str(i) for i in range(self.n_channels)]
        self.channels_locs = [np.array([0,0,0]) for i in range(self.n_channels)]
        self.channels_locs = np.array(self.channels_locs)

    def Filter(self, filter_type, wp, ws, gpass, gstop):
        N, critical_points = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=self.Fs)
        sos = signal.iirfilter(N, critical_points, btype=filter_type, analog=False, ftype='butter', output='sos', fs=self.Fs)
        Filtered_series = self.data.copy()
        for i in range(self.n_channels):
            Filtered_series[:,i] = signal.sosfiltfilt(sos, self.data[:,i], axis=- 1, padtype=None)
        Filtered = Raw(Filtered_series,self.Fs)
        Filtered.unit = self.unit
        Filtered.channels_names = self.channels_names
        Filtered.channels_locs = self.channels_locs
        return Filtered
    
    def select_channels_by_index(self,channel_list):
        selected = Raw(self.data[:,channel_list],self.Fs)
        selected.unit = self.unit
        for ch in range(len(channel_list)):
            selected.channels_names[ch] = self.channels_names[channel_list[ch]]
            selected.channels_locs[ch] = self.channels_locs[channel_list[ch]]
        return selected
    
    def select_channels_by_name(self,channel_list):
        channel_indexs = []
        for i in channel_list:
            channel_indexs.append(self.channels_names.index(i))
        selected = Raw.select_channels_by_index(self,channel_indexs)
        return selected
    
    def scale(self,gain,drift):
        self.data = self.data*gain+drift
        
    def set_unit(self,unit):
        self.unit = unit
        
    def change_unit(self,ori_unit,new_unit):
        unitlist = ['uV','mV','V']
        gain = pow(1000,(unitlist.index(new_unit)-unitlist.index(ori_unit)))
        self = self.scale(gain,0)
        self.unit = new_unit
    
    def get_LFP(self):
        LFP = self.Filter('highpass',1,0.1,3,60)
        LFP = LFP.Filter('lowpass',300,1000,3,60)
        return LFP
    
    def get_spikes(self):
        spike_traces = self.Filter('bandpass',[300,6000],[100,10000],3,60)
        return spike_traces
    
    def extract_by_median(self,k):
        data = self.data
        absolute = np.abs(data)
        Fs = self.Fs
        med = np.median(absolute,axis=0)
        thres = med/0.6745*k
        time_points=[]
        waveforms=[]
        for i in range(self.n_channels):
            samp_points,_ = signal.find_peaks(absolute[:,i],height=thres[i],distance=Fs*0.002)
            for samp_point in samp_points:
                waveform = data[int(np.floor(samp_point-Fs/1000)):int(np.ceil(samp_point+Fs/1000)),i]
                if samp_point == samp_points[0]:
                    waves = np.array([waveform])
                else:
                    waves = np.append(waves,[waveform],axis=0)
            time_points += [samp_points/Fs]
            waveforms += [waves]
        return Detect(waveforms,time_points,Fs,self.time_length,self.unit,self.channels_names,self.channels_locs)

    def set_channels_names(self,channels_names):
        self.channels_names = channels_names
        if type(self.channels_names) is not list:
            raise ValueError('通道名称必须为列表')
        if len(self.channels_names) != self.n_channels:
            raise ValueError('通道名称列表长度必须与通道数一致')
        
    def set_channels_locs(self,locs):
        self.channels_locs = locs
        if (type(self.channels_locs) is not np.ndarray) or (np.shape(self.channels_locs) != (self.n_channels,3)):
            raise ValueError('通道坐标必须为通道数*3的numpy.array')
            
    def sort_by(self,sortername):
        neurons = ddlsorters.Sorter(self,sortername)
        return neurons