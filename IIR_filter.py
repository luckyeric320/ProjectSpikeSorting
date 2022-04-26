from scipy import signal
import numpy as np

class Raw:
    
    def __init__(self,time_series,Fs):
        self.data = time_series
        self.n_channels = np.shape(time_series)[1]
        self.Fs = Fs

    def Filter(self, filter_type, wp, ws, gpass, gstop):
        N, critical_points = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=self.Fs)
        #print('N='+str(N))
        sos = signal.iirfilter(N, critical_points, btype=filter_type, analog=False, ftype='butter', output='sos', fs=self.Fs)
        Filtered_series = self.data.copy()
        for i in range(self.n_channels):
            Filtered_series[:,i] = signal.sosfiltfilt(sos, self.data[:,i], axis=- 1, padtype=None)
        Filtered = Raw(Filtered_series,self.Fs)
        return Filtered
    
    def get_n_channels(self):
        return self._channels
    
    def get_time_length(self):
        return np.shape(self.data)[0]/self.Fs
    
    def select_channels(self,channel_list):
        selected = Raw(self.data[:,channel_list],self.Fs)
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
