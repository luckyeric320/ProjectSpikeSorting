
#! /usr/bin/python3.7
# -- coding: utf-8 -- **

import sys
import os
import time
import nibabel as nib #对常见的医学和神经影像文件格式进行读写
import numpy as np
from surfer import Brain #用于可视化大脑皮层表面的三维网格上绘制数据
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction
from tvtk.util.ctf import PiecewiseFunction

import mne
from sklearn import preprocessing  
import scipy
from scipy import signal
from scipy.signal import convolve2d,filtfilt,iirnotch,butter,resample_poly
import zipfile
import seaborn as sns

#@mlab.animate(delay=200)

def compute_hfer(target_data, base_data, fs):
    target_sq = target_data ** 2
    base_sq = base_data ** 2
    window = int(fs / 2.0)
    target_energy=convolve2d(target_sq,np.ones((1,window)),'same')
    base_energy=convolve2d(base_sq,np.ones((1,window)),'same')
    base_energy_ref = np.sum(base_energy, axis=1) / base_energy.shape[1]
    target_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, target_energy.shape[1]))
    base_de_matrix = base_energy_ref[:, np.newaxis] * np.ones((1, base_energy.shape[1]))
    norm_target_energy = target_energy / target_de_matrix.astype(np.float32)
    norm_base_energy = base_energy / base_de_matrix.astype(np.float32)
    return norm_target_energy, norm_base_energy


def featureExtraction():    
    edf_data = mne.io.read_raw_edf("./S1_ictal.edf", preload=True, stim_channel=None)
    #edf_data.plot()
    fs = 500
    valid_chns_index=np.arange(len(edf_data.ch_names))
    valid_chns=np.array(edf_data.ch_names)
    modified_edf_data = edf_data[valid_chns_index,:][0][:,:]
    #preprocessing
    modified_edf_data = resample_poly(modified_edf_data, 500, 2000, axis=1)
    modified_edf_data=modified_edf_data-np.mean(modified_edf_data,axis=0)
    #notch filter
    notch_freqs=np.arange(50,151,50)
    for nf in notch_freqs:
        tb,ta=iirnotch(nf/(fs/2),30)
        modified_edf_data=filtfilt(tb,ta,modified_edf_data,axis=-1)
    #band filter
    band_low = 60
    band_high = 140
    nyq = fs/2
    b, a = butter(5, np.array([band_low/nyq, band_high/nyq]), btype = 'bandpass')
    modified_edf_data = filtfilt(b,a,modified_edf_data)
    
    baseline_data=modified_edf_data[:,:500]
    target_data=modified_edf_data[:,500:35500]
    
    batch_data,base_e=compute_hfer(baseline_data, target_data, fs)
    #不知道为啥，反过来比较好看
    batch_data = np.flip(batch_data,1)
    sns.heatmap(batch_data)
    
    np.save("./HFData.npy", batch_data)
    #return batch_data
    
def visualization(dir,hf_file,outdir,result):
    HF_Data = np.load(hf_file,allow_pickle=True)
    # print(os.path.join(dir,"HFData.npy"))
    X_minMax = np.array(HF_Data[:128])
    
    elecs_xyzDict=np.load(os.path.join(dir,"chnXyzDict.npy"),allow_pickle=True)[()] #电极点坐标
    #根据电极坐标筛选seeg数据
    elec_chname = []
    elec_xyz =np.empty([0,3])
    elec_dict = {}
    for k1, v1 in elecs_xyzDict.items():#访问每个电极点的坐标

        elec_name = k1
        elec_info = v1
        elec_xyz = np.concatenate((elec_xyz,elec_info),axis=0)
        for ei in range(len(elec_info)):
            elec_chname.append(elec_name+str(ei+1))
            elec_dict[elec_name+str(ei+1)] = elec_info[ei]

    brain_data=nib.load(os.path.join(dir,"example/mri/orig.mgz")) #皮层重建结果
    #brain_data.orthoview()
    aff_matrix=brain_data.header.get_affine() #仿射矩阵
    #print(aff_matrix)
    verl,facel=nib.freesurfer.read_geometry(os.path.join(dir,"example/surf/lh.pial")) #左脑皮层表面
    verr,facer=nib.freesurfer.read_geometry(os.path.join(dir,"example/surf/rh.pial")) #右脑皮层表面
    #ver:顶点坐标xyz；face：网格三角形（mesh triangle）三元组，每个元组表示一个面中三个顶点的序号（索引）
    all_ver=np.concatenate([verl,verr],axis=0)#所有顶点
    tmp_facer=facer+verl.shape[0]
    all_face=np.concatenate([facel,tmp_facer],axis=0)
    opacity=0.4
    ambient=0.4225
    specular = 0.3
    specular_power = 20
    diffuse = 0.5
    interpolation='phong'
    mlab.options.offscreen = True
    mlab.figure(bgcolor=(0.8,0.8,0.8),size=(1500,1500))
    figure=mlab.gcf()

    mesh=mlab.triangular_mesh(all_ver[:,0],all_ver[:,1],all_ver[:,2],all_face,color=(1.,1.,1.),representation='surface',opacity=opacity,line_width=1.)
    # change OpenGL mesh properties for phong point light shading
    mesh.actor.property.ambient = ambient
    mesh.actor.property.specular = specular
    mesh.actor.property.specular_power = specular_power
    mesh.actor.property.diffuse = diffuse
    mesh.actor.property.interpolation = interpolation
    mesh.actor.property.backface_culling = True
    if opacity <= 1.0:
        mesh.scene.renderer.trait_set(use_depth_peeling=True)  # , maximum_number_of_peels=100, occlusion_ratio=0.4)
    # Make the mesh look smoother
    for child in mlab.get_engine().scenes[0].children:
        poly_data_normals = child.children[0]
        poly_data_normals.filter.feature_angle = 80.0    
        
    pts = mlab.points3d(elec_xyz[:,0], elec_xyz[:,1], elec_xyz[:,2],X_minMax[:,0], scale_factor=5)

    @mlab.animate(delay=10)
    def anim(f):
        ms = pts.mlab_source
        for i in range(10):
            scalars = X_minMax[:,i+1]
            print(i)
            ms.reset(scalars=scalars)
            if i < 10:
                pts.module_manager.source.save_output(os.path.join(outdir, '0' + str(i) + '.vtk'))
            else:
                pts.module_manager.source.save_output(os.path.join(outdir, str(i) + '.vtk'))

            if i < 10:
                f.write(os.path.join(outdir,'0'+str(i)+'.vtk'),str(i)+'.vtk')
            else:
                f.write(os.path.join(outdir,str(i)+'.vtk'),str(i)+'.vtk')


    try:
        f = zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED)
        anim(f)
        f.close()
    except Exception as e:
        f.close()
        print(e)



    


