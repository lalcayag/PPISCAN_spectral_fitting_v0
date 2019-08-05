# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:13:10 2018

@author: lalc
"""

import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize
from os import listdir
import tkinter as tkint
import tkinter.filedialog
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import ppiscanprocess.filtering as fl
import pickle

import numpy as np
import pandas as pd
import scipy as sp

import importlib

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# In[]

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

fmt = FormatScalarFormatter("%.2f")    

from mpl_toolkits.axes_grid1 import make_axes_locatable 
from matplotlib.ticker import LinearLocator, FormatStrFormatter

f = 24

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])

# In[]

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Input dir')
root.destroy()

root = tkint.Tk()
file_out_path = tkint.filedialog.askdirectory(parent=root,title='Choose an Output dir')
root.destroy()

cwd = os.getcwd()
os.chdir(file_in_path)

# In[column labels]

iden_lab = np.array(['num1','num2','start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab

#Labels for range gates and speed
vel_lab = np.array(['range_gate','ws','CNR','Sb'])

for i in np.arange(99):

    labels = np.concatenate((labels,vel_lab))
    
# In[]
    
filelist_s = [(filename,getsize(join(file_in_path,filename)))
             for filename in listdir(file_in_path) if getsize(join(file_in_path,filename))>1000]
size_s = list(list(zip(*filelist_s))[1])
filelist_s = list(list(zip(*filelist_s))[0])

# In[Different eatres of the DBSCAN filter]

#feat0 = ['range_gate','ws']
#feat1 = ['range_gate','ws','CNR']
feat2 = ['range_gate','ws','dvdr']
#feat3 = ['range_gate','ws','CNR','dvdr']
#feat4 = ['range_gate','azim','ws']
feat5 = ['range_gate','azim','dvdr']
feat6 = ['range_gate','azim','ws','dvdr']
    
# In[Geometry of scans]

r_0 = np.linspace(150,7000,198) # It is 105!!!!!!!!!!!!!!!!!!!!!!!!!
r_1 = np.linspace(150,7000,198)
phi_0 = np.linspace(256,344,45)*np.pi/180
phi_1 = np.linspace(196,284,45)*np.pi/180
r_0_g, phi_0_g = np.meshgrid(r_0,phi_0)
r_1_g, phi_1_g = np.meshgrid(r_1,phi_1)
  
# In[]
######################## Identifying noise
ae = [0.025, 0.05, 0.075]
L = [125,250,500,750]
G = [0,1,2,2.5,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)
utot = np.linspace(15,25,5)
Dir = np.linspace(90,270,5)*np.pi/180
onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]
ind_noise = []
param = []
sigma = []
for dir_mean in Dir:
    for u_mean in utot:
        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
            vlos0_file = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            #vlos1_file = 'vlos1'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            vlos0_noise_file = 'noise0_'+vlos0_file
            #vlos1_noise_file = 'noise1_'+vlos1_file
            if vlos0_file in onlyfiles:
                #v_in.append(vlos0_file) 
                param.append(np.array([ae_i,L_i,G_i,seed_i,u_mean,int(dir_mean*180/np.pi)]))
                vlos0 = np.fromfile(vlos0_file, dtype=np.float32)
                sigma.append(np.std(vlos0))
                vlos0_noise = np.fromfile(vlos0_noise_file, dtype=np.float32)
                diff = vlos0-vlos0_noise
                ind_noise.append(np.reshape(diff != 0,r_0_g.shape))
param = np.array(param)
#### Figures   
                
scan = 10
ind_scan = df_noise.scan == scan

f=16
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_noise.CNR.loc[ind_scan].values, 50,cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$CNR, simulated$", fontsize=f)
ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=32,verticalalignment='top')

scan = 1000
ind_scan = df.scan == scan

f=16
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df.CNR.loc[ind_scan].values, 50,cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$V_{LOS}\:[m/s]$", fontsize=f)
ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')

f=16
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df.ws.loc[ind_scan].values, np.linspace(-8,8,50),cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$V_{LOS}\:[m/s]$", fontsize=f)
ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')




f=16
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_noise.ws.loc[ind_scan].values, 50,cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$V_{LOS}\:[m/s]$", fontsize=f)
ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=32,verticalalignment='top')

############################

#plt.scatter((r_0_g*np.cos(phi_0_g))[i_noise],(r_0_g*np.sin(phi_0_g))[i_noise])
#plt.colorbar()
#
#plt.figure()
#plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_vlos0_noise_median.ws.loc[ind_scan ].values, 30, cmap='jet')
#plt.colorbar()

################## DBSCAN filter, commented are not good, just for comparison

with open('df_syn_fin.pkl', 'rb') as reader:
    df_noise = pickle.load(reader)     
    
t_step = 3 #5
ind=np.unique(df_noise.scan.values)%t_step==0
times= np.unique(np.append(np.unique(df_noise.scan.values)[ind],df_noise.scan.values[-1])) 
#mask0=pd.DataFrame()
#mask1=pd.DataFrame()
#mask2=pd.DataFrame()
#mask3=pd.DataFrame()
#mask4=pd.DataFrame()
#mask5=pd.DataFrame()
mask6=pd.DataFrame()

for i in range(len(times)-1):
       print(times[i])
       loc = (df_noise.scan>=times[i]) & (df_noise.scan<times[i+1])
#       mask0 = pd.concat([mask0,fl.data_filt_DBSCAN(df_noise.loc[loc],feat0)]) 
#       mask1 = pd.concat([mask1,fl.data_filt_DBSCAN(df_noise.loc[loc],feat1)]) 
#       mask2 = pd.concat([mask2,fl.data_filt_DBSCAN(df_noise.loc[loc],feat2)])
#       mask3 = pd.concat([mask3,fl.data_filt_DBSCAN(df_noise.loc[loc],feat3)]) 
#       mask4 = pd.concat([mask4,fl.data_filt_DBSCAN(df_noise.loc[loc],feat4)]) 
#       mask5 = pd.concat([mask5,fl.data_filt_DBSCAN(df_noise.loc[loc],feat5)])
       mask6 = pd.concat([mask6,fl.data_filt_DBSCAN(df_noise.loc[loc],feat6,epsCNR=True)])        
       if i == range(len(times)-1):
          loc = df_noise.scan == times[i+1]
#          mask0 = pd.concat([mask0,fl.data_filt_DBSCAN(df_noise.loc[loc],feat0)]) 
#          mask1 = pd.concat([mask1,fl.data_filt_DBSCAN(df_noise.loc[loc],feat1)]) 
#          mask2 = pd.concat([mask2,fl.data_filt_DBSCAN(df_noise.loc[loc],feat2)])
#          mask3 = pd.concat([mask3,fl.data_filt_DBSCAN(df_noise.loc[loc],feat3)]) 
#          mask4 = pd.concat([mask4,fl.data_filt_DBSCAN(df_noise.loc[loc],feat4)]) 
#          mask5 = pd.concat([mask5,fl.data_filt_DBSCAN(df_noise.loc[loc],feat5)])
          mask6 = pd.concat([mask3,fl.data_filt_DBSCAN(df_noise.loc[loc],feat6,epsCNR=True)])           

################### 5 scans steps
          
#with open('mask_filter_r_ws_5.pkl', 'wb') as writer:
#    pickle.dump(mask0,writer)
#with open('mask_filter_r_ws_CNR_5.pkl', 'wb') as writer:
#    pickle.dump(mask1,writer)
#with open('mask_filter_r_ws_dvdr_5.pkl', 'wb') as writer:
#    pickle.dump(mask2,writer)
#with open('mask_filter_r_ws_CNR_dvdr_5.pkl', 'wb') as writer:
#    pickle.dump(mask3,writer)
#   
#with open('mask_filter_r_azim_ws_5.pkl', 'wb') as writer:
#    pickle.dump(mask4,writer)
#with open('mask_filter_r_azim_dvdr_5.pkl', 'wb') as writer:
#    pickle.dump(mask5,writer)  
#with open('mask_filter_r_azim_ws_dvdr_5.pkl', 'wb') as writer:
#    pickle.dump(mask6,writer) 
     
#######################3 scans steps
          
#with open('mask_filter_r_ws_3.pkl', 'wb') as writer:
#    pickle.dump(mask0,writer)
#with open('mask_filter_r_ws_CNR_3.pkl', 'wb') as writer:
#    pickle.dump(mask1,writer)
#with open('mask_filter_r_ws_dvdr_3.pkl', 'wb') as writer:
#    pickle.dump(mask2,writer)
#with open('mask_filter_r_ws_CNR_dvdr_3.pkl', 'wb') as writer:
#    pickle.dump(mask3,writer)
#with open('mask_filter_r_azim_ws_3.pkl', 'wb') as writer:
#    pickle.dump(mask4,writer)
#with open('mask_filter_r_azim_dvdr_3.pkl', 'wb') as writer:
#    pickle.dump(mask5,writer)  
#with open('mask_filter_r_azim_ws_dvdr_3.pkl', 'wb') as writer:
#    pickle.dump(mask6,writer)   
  
####################### Loading results
          

with open('mask_filter_r_ws_3.pkl', 'rb') as reader:
    mask0 = pickle.load(reader)
with open('mask_filter_r_ws_CNR_3.pkl', 'rb') as reader:
    mask1 = pickle.load(reader)
with open('mask_filter_r_ws_dvdr_3.pkl', 'rb') as reader:
    mask2 = pickle.load(reader)
with open('mask_filter_r_ws_CNR_dvdr_3.pkl', 'rb') as reader:
    mask3 = pickle.load(reader)
with open('mask_filter_r_azim_ws_3.pkl', 'rb') as reader:
    mask4 = pickle.load(reader)
with open('mask_filter_r_azim_dvdr_3.pkl', 'rb') as reader:
    mask5 = pickle.load(reader)  
with open('mask_filter_r_azim_ws_dvdr_3.pkl', 'rb') as reader:
    mask6 = pickle.load(reader) 
    
#######################Filtering of dataFrame, different parameters
    
r_0 = np.linspace(150,7000,198) # It is 105!!!!!!!!!!!!!!!!!!!!!!!!!
r_1 = np.linspace(150,7000,198)
phi_0 = np.linspace(256,344,45)*np.pi/180
phi_1 = np.linspace(196,284,45)*np.pi/180
r_0_g, phi_0_g = np.meshgrid(r_0,phi_0)
r_1_g, phi_1_g = np.meshgrid(r_1,phi_1)
    
#df_noise0 = df_noise.copy() 
#df_noise1 = df_noise.copy() 
#df_noise2 = df_noise.copy() 
#df_noise3 = df_noise.copy() 
#df_noise4 = df_noise.copy() 
#df_noise5 = df_noise.copy() 
df_noise6 = df_noise.copy()    

#df_noise0.ws = df_noise0.ws.mask(mask0)
#df_noise1.ws = df_noise1.ws.mask(mask1)
#df_noise2.ws = df_noise2.ws.mask(mask2)
#df_noise3.ws = df_noise3.ws.mask(mask3)
#df_noise4.ws = df_noise4.ws.mask(mask4)
#df_noise5.ws = df_noise5.ws.mask(mask5)
df_noise6.ws = df_noise6.ws.mask(mask6)

############################### Median filter, parameter exploration and analysis on stats

n_w = np.arange(2,10,1)
m_w = np.arange(2,10,1)
lim = np.linspace(1,8,8)

n_w, m_w, lim = np.meshgrid(n_w,m_w, lim)
ind_noise_array = np.vstack(tuple(ind_noise))

noise_det = []
not_noise_det = []
reliable_scan = []

chunk_size = 45
counting = 512
for nw, mw, l in zip(n_w.flatten()[512-counting:], m_w.flatten()[512-counting:], lim.flatten()[512-counting:]):
    df_noise_median_i = fl.data_filt_median(df_noise,lim_m=l,lim_g=100,n=nw, m=mw)
    reliable = ~np.isnan(df_noise_median_i.ws.values) & ~ind_noise_array
    n = np.isnan(df_noise_median_i.ws.values) & ind_noise_array
#    nn = np.isnan(df_noise_median_i.ws.values) & ~ind_noise_array
    reliable_scan.append([np.sum(reliable[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])
    noise_det.append([np.sum(n[i:i+chunk_size,:])/np.sum(ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])
#    not_noise_det.append([np.sum(nn[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])
    counting-=1
    print(counting)
############################## Clustering filter stats
#noise2 = np.isnan(df_noise2.ws.values) & ind_noise_array
#noise_removed2 = np.array([np.sum(noise2[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])   
#noise5 = np.isnan(df_noise5.ws.values) & ind_noise_array
#noise_removed5 = np.array([np.sum(noise5[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])   
noise6 = np.isnan(df_noise6.ws.values) & ind_noise_array
noise_removed6 = np.array([np.sum(noise6[i:i+chunk_size,:])/np.sum(ind_noise_array[i:i+chunk_size,:]) for i in range(0, df_noise6.ws.values.shape[0], chunk_size)])   

#reliable2 = ~np.isnan(df_noise2.ws.values) & ~ind_noise_array 
#reliable_scan2 = np.array([np.sum(reliable2[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])   
#reliable5 = ~np.isnan(df_noise5.ws.values) & ~ind_noise_array 
#reliable_scan5 = np.array([np.sum(reliable5[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, n.shape[0], chunk_size)])   
reliable6 = ~np.isnan(df_noise6.ws.values) & ~ind_noise_array 
reliable_scan6 = np.array([np.sum(reliable6[i:i+chunk_size,:])/np.sum(~ind_noise_array[i:i+chunk_size,:]) for i in range(0, df_noise6.ws.values.shape[0], chunk_size)])   

#with open('reliable_scan6.pkl', 'wb') as writer:
#    pickle.dump(reliable_scan6,writer)
#with open('reliable_scan2.pkl', 'wb') as writer:
#    pickle.dump(reliable_scan2,writer)  
#with open('reliable_scan_sensitivity_median_nm.pkl', 'wb') as writer:
#    pickle.dump(reliable_scan,writer)
#with open('noise_det_sensitivity_median_nm.pkl', 'wb') as writer:
#    pickle.dump(noise_det,writer)
#with open('not_noise_det_sensitivity_median.pkl', 'wb') as writer:
#    pickle.dump(not_noise_det,writer)

with open('reliable_scan_sensitivity_median_nm.pkl', 'rb') as reader:
    reliable_scan = pickle.load(reader)
    
with open('noise_det_sensitivity_median_nm.pkl', 'rb') as reader:
    noise_det = pickle.load(reader)

noise_weight = np.array([np.sum(ind_noise_array[i:i+chunk_size,:])/len(ind_noise_array[i:i+chunk_size,:].flatten()) for i in range(0, ind_noise_array.shape[0], chunk_size)])

mean_noise_det = np.reshape(np.array([np.mean(nd) for nd in noise_det]),n_w.shape)   
#mean_not_noise_det = np.reshape(np.array([np.mean(nd) for nd in not_noise_det]),n_w.shape)  
mean_reliable = np.reshape(np.array([np.mean(rel) for rel in reliable_scan]),n_w.shape)   

std_noise_det = np.reshape(np.array([np.std(nd) for nd in noise_det]),n_w.shape)     
std_reliable = np.reshape(np.array([np.std(rel) for rel in reliable_scan]),n_w.shape)   

std_tot = np.reshape(np.array([np.std(.5*(rel+nd)) for rel,nd in zip(np.array(reliable_scan),np.array(noise_det))]),n_w.shape)

mean_tot_w = np.reshape(np.array([np.mean((rel*(1-noise_weight)+nd*noise_weight)) for rel,nd in zip(np.array(reliable_scan),np.array(noise_det))]),n_w.shape)
    
#mean_tot_6 = np.mean(.5*(reliable_scan6[:-1] + noise_removed6[:-1]))
#std_tot_6 = np.std(.5*(reliable_scan6[:-1] + noise_removed6[:-1]))

mean_tot_6_w = np.mean((reliable_scan6[:-1]*(1-noise_weight[:-1]) + noise_removed6[:-1]*noise_weight[:-1]))


############ Some figures

l = 2

plt.figure()
plt.contourf(n_w[:,:,l],m_w[:,:,l],mean_noise_det[:,:,l]/np.mean(noise_removed6[:-1]),30,cmap='jet')
plt.colorbar()

for l in range(len(np.linspace(1,8,8))):
    fig, ax = plt.subplots()
    im = ax.contourf(n_w[:,:,l],m_w[:,:,l],(mean_tot_w[:,:,l]),30,cmap='jet')
    ax.set_title('$\Delta V_{LOS,\:threshold}\:=\:'+str(np.mean(lim[:,:,l]))+'\:m/s$', fontsize=16)
    ax.set_xlabel('$Radial\:window\:length,\: n_r$', fontsize=16)
    ax.set_ylabel('$Azimuth\:window\:length,\: n_{\phi}$', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
    cbar.ax.set_ylabel("$\eta_{tot}$", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax.tick_params(labelsize=16)
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=32,verticalalignment='top')
    fig.tight_layout()

l = lim.flatten()
nw = n_w.flatten()/l
mw = m_w.flatten()/l
tot = (mean_tot_w).flatten()
noise = (mean_noise_det).flatten()
rec = (mean_reliable).flatten()
#################################
def mov_ave(mylist,N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return np.array(moving_aves)
##################################Figure eta curve
indexnw = np.argsort(nw)
indexmw = np.argsort(mw)
indexnmw = np.argsort(nw*mw*l)

totm = mov_ave(tot[indexnmw], 40)
noisem = mov_ave(noise[indexnmw], 40)
recm = mov_ave(rec[indexnmw], 40)
off = -totm.shape[0]+tot[indexnmw].shape[0]

fig, ax = plt.subplots()
ax.plot((nw*mw*l)[indexnmw][off:],totm, label = '$\eta_{tot}$', c = 'k', lw=2)

ax.scatter((nw*mw*l)[indexnmw],tot[indexnmw],s = 2, c = 'k', alpha = .2)

ax.plot((nw*mw*l)[indexnmw][off:],noisem, label = '$\eta_{noise}$', c = 'r', lw=2)

ax.scatter((nw*mw*l)[indexnmw], noise[indexnmw],s = 2, c = 'r', alpha = .2)

ax.plot((nw*mw*l)[indexnmw][off:],recm, label = '$\eta_{rec}$', c = 'b', lw=2)

ax.scatter((nw*mw*l)[indexnmw],rec[indexnmw],s = 2, c = 'b', alpha = .2)

ax.set_xscale('log')
ax.set_xlabel('$n_rn_\phi/\Delta V_{LOS,threshold}$',fontsize = 16)
ax.set_ylabel('$\eta$',fontsize = 16)
ax.legend(fontsize = 16)
ax.tick_params(labelsize = 16)
ax.set_xlim(1,100)
fig.tight_layout()

plt.figure()
plt.scatter((nw*mw)[indexnmw],tot[indexnmw])
plt.xscale('log')

plt.figure()
plt.scatter((nw*mw)[indexnmw],noise[indexnmw])
plt.xscale('log')

plt.figure()
plt.scatter((nw*mw)[indexnmw],rec[indexnmw])
plt.xscale('log')

plt.plot(mw[indexmw],tot[indexmw])
plt.plot((nw*mw*l)[indexnmw],tot[indexnmw])

plt.scatter(nw, mw, c = tot)

plt.figure()
plt.contourf(n_w,lim,mean_reliable/np.mean(reliable_scan6[:-1]),30,cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(n_w,lim,std_tot/std_tot_6,30,cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(n_w,lim,(mean_tot/mean_tot_6),30,cmap='jet')
plt.colorbar()

plt.figure()
plt.contourf(n_w,lim,(mean_tot/mean_tot_6)*(std_tot_6/std_tot),30,cmap='jet')
plt.colorbar()

############################################### Figures

fig, ax = plt.subplots()
ax.hist(noise_weight,bins=30,histtype='step',lw=2, color = 'k')
ax.set_xlabel('$f_{noise}$', fontsize=16)
ax.set_ylabel('$Counts$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=24,verticalalignment='top')
fig.tight_layout()


indexes = np.argsort((mean_tot_w/mean_tot_6_w).flatten())[::-1]
plt.figure()
plt.plot((mean_tot_w/mean_tot_6_w).flatten()[indexes])
plt.title('meantot')
plt.figure()
plt.plot((m_w*n_w/lim**2).flatten()[indexes]/(np.max((m_w*n_w/lim))))
plt.title('mw*nw/lim')
plt.figure()
plt.plot(mean_noise_det.flatten()[indexes])
plt.title('noisedet')
plt.figure()
plt.plot(mean_reliable.flatten()[indexes])
plt.title('noisedet')


plt.figure()
plt.plot((m_w/lim).flatten()[indexes]/(np.max((m_w/lim))))
plt.title('mw/lim')

plt.figure()
plt.plot((n_w/lim).flatten()[indexes]/(np.max((n_w/lim))))
plt.title('nw/lim')

#
#plt.figure()
#plt.plot((noise_det/np.mean(noise_removed6)).flatten()[indexes][:500])
#plt.title('noisew')
#plt.figure()
#plt.plot((reliable_scan/np.mean(reliable_scan6)).flatten()[indexes][:500])
#plt.title('relw')
#plt.figure()
#plt.plot((noise_weight).flatten()[indexes][:500])
#plt.title('noise')
#plt.figure()
#plt.plot((lim).flatten()[indexes][:500])
#plt.title('lim')
#plt.figure()
#plt.plot((m_w).flatten()[indexes][:500])
#plt.title('mw')
#plt.figure()
#plt.plot((n_w).flatten()[indexes][:500])
#plt.title('nw')
#plt.figure()
#plt.plot((param[:,0]).flatten()[indexes][:500])
#plt.title('ae')
#plt.figure()
#plt.plot((param[:,1]).flatten()[indexes][:500])
#plt.title('L')
#plt.figure()
#plt.plot((param[:,2]).flatten()[indexes][:500])
#plt.title('G')
#plt.figure()
#plt.plot((param[:,3]).flatten()[indexes][:500])
#plt.title('U')
#plt.figure()
#plt.plot((param[:,4]).flatten()[indexes][:500])
#plt.title('dir')



plt.figure()
plt.plot((((m_w*n_w/lim).flatten())/param[:,0])[indexes][:500])
plt.title('mw')

for i in range(5):
    i = 3    
    f = 20
    indr = reliable_scan6[:-1]>.78
    fig, ax = plt.subplots()
    ax.hist(noise_removed6[:-1][indr],bins=30, histtype = 'step', label = 'Clustering', lw = 2, color = 'r')
    ax.hist(noise_det[indexes[i]][:-1],bins=30, histtype = 'step', label = 'Median', lw = 2, color = 'k')
    fig.legend(loc = (.6,.75),fontsize = f)
    ax.tick_params(axis='both', which='major', labelsize = f)
    ax.set_xlabel('$\eta_{noise}$',fontsize = f)
    ax.set_ylabel('$Counts$',fontsize = f)
#    ax.set_xlim([.3,1.])
#    ax.set_ylim([.0,140])
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=28,
        verticalalignment='top')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(reliable_scan6[:-1][indr],bins=30, histtype = 'step', label = 'Clustering', lw = 2, color = 'r')
    ax.hist(reliable_scan[indexes[i]][:-1],bins=30, histtype = 'step', label = 'Median', lw = 2, color = 'k')
    fig.legend(loc = (.18,.65),fontsize=f)
    ax.tick_params(axis='both', which='major', labelsize=f)
    ax.set_xlabel('$\eta_{recov}$',fontsize = f)
    ax.set_ylabel('$Counts$',fontsize = f)
#    ax.set_xlim([.85,1.])
#    ax.set_ylim([.0,180])
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=28,
        verticalalignment='top')
    fig.tight_layout()
    
    fig, ax = plt.subplots()
    ax.hist(reliable_scan6[:-1][indr]*(1-noise_weight[:-1][indr])+noise_removed6[:-1][indr]*noise_weight[:-1][indr],bins=30, histtype = 'step', label = 'Clustering', lw = 2, color = 'r')
    ax.hist(reliable_scan[indexes[i]][:-1]*(1-noise_weight[:-1])+noise_det[indexes[i]][:-1]*noise_weight[:-1],bins=30, histtype = 'step', label = 'Median', lw = 2, color = 'k')
    fig.legend(loc = (.18,.65),fontsize=f)
    ax.tick_params(axis='both', which='major', labelsize=f)
    ax.set_xlabel('$\eta_{tot}$',fontsize = f)
    ax.set_ylabel('$Counts$',fontsize = f)
#    ax.set_xlim([.7,1.])
#    ax.set_ylim([.0,130])
    ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=28,
        verticalalignment='top')
    fig.tight_layout()


fig, ax = plt.subplots()
#plt.hist(noise_removed2[:-1],bins=30,alpha=.5,label = 'Clustering filter0')
##plt.hist(noise_removed5[:-1],bins=30,alpha=.5,label='5'+str(np.mean(noise_removed5)))
ax.hist(.5*(noise_removed6[:-1]+reliable_scan6[:-1]),bins=30, histtype = 'step', label = 'Clustering filter', lw = 2)
ax.hist(.5*(np.array(noise_det[19][:-1])+np.array(reliable_scan[19][:-1])),bins=30, histtype = 'step', label = 'Median filter', lw = 2)
fig.legend(loc = (.6,.75),fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('$(\eta_{recov}+\eta_{noise})/2$',fontsize = 16)
ax.set_ylabel('$Counts$',fontsize = 16)



###############################
# Noise figures

noise1 = np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.5,azim_frac=.3, tot = 'yes'),r_0_g.shape)
noise1 = noise1+np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.7,azim_frac=.6, tot = 'yes'),r_0_g.shape)
noise1 = noise1+np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.9,azim_frac=.9, tot = 'yes'),r_0_g.shape)

ind = noise1==0

a = np.max(noise1)
c = np.min(noise1)
if (a-c) > 0:
    b = 1
    d = -1   
    m = (b - d) / (a - c)
noise1 = (m * (noise1 - c)) + d
noise1 = noise1*35.0

noise1[ind] = np.nan

f=16
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
plt.contourf(r_0_g,phi_0_g, noise1, 50,cmap='jet')
#ax.set_ylabel('X (From west to east) [m]', fontsize=12, weight='bold')
#ax.set_xlabel('Y (From north to south) [m]', fontsize=12, weight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))

cbar.ax.tick_params(labelsize=f)
ax.tick_params(labelsize=f)

cbar.ax.set_ylabel("$V_{LOS}\:[m/s]$", fontsize=f)
ax.set_ylabel('$x\:[m]$', fontsize=f, weight='bold')
ax.set_xlabel('$y\:[m]$', fontsize=f, weight='bold')
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=32,verticalalignment='top')


#############################################

with open('noise_removed2.pkl', 'wb') as writer:
    pickle.dump(noise_removed2,writer)
with open('not_noise_removed2.pkl', 'wb') as writer:
    pickle.dump(not_noise_removed2,writer)    
with open('noise_removed5.pkl', 'wb') as writer:
    pickle.dump(noise_removed5,writer)
with open('not_noise_removed5.pkl', 'wb') as writer:
    pickle.dump(not_noise_removed5,writer)  
with open('noise_removed6.pkl', 'wb') as writer:
    pickle.dump(noise_removed6,writer)
with open('not_noise_removed6.pkl', 'wb') as writer:
    pickle.dump(not_noise_removed6,writer)    
with open('noise_removed_med.pkl', 'wb') as writer:
    pickle.dump(noise_removed_med,writer)
with open('not_noise_removed_med.pkl', 'wb') as writer:
    pickle.dump(not_noise_removed_med,writer)

tot_noise = np.array([np.sum(ind_noise[i]) for i in range(len(ind_noise[:-1]))])

tot_not_noise = np.array([np.sum(~ind_noise[i]) for i in range(len(ind_noise[:-1]))])

noise_diff = np.array([(clust-med)/med for med, clust in zip(noise_removed_med[:-1],noise_removed6[:-1])])

not_noise_diff = np.array([(clust-med)/med for med, clust in zip(not_noise_removed_med[:-1],not_noise_removed6[:-1])])

plt.figure()
plt.hist(noise_diff,bins=30,alpha=.5,label=str(np.mean(noise_diff)))
plt.legend()

plt.figure()
plt.hist(not_noise_diff,bins=30,alpha=.5,label=str(np.mean(not_noise_diff)))
plt.legend()

 # In[]
p = ws3_w_df.elev.unique()
r = np.array(ws3_w_df.iloc[(ws3_w_df.elev==
                               min(p)).nonzero()[0][0]].range_gate)
r_g, p_g = np.meshgrid(r, np.radians(p)) # meshgrid

scan_n = 1

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(377):
    ax.cla()
    ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df.ws.loc[df.scan==scan_n].values,100,cmap='jet')
    plt.pause(.001)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(375):
    ax.cla()
    ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df_int.ws.loc[scan_n].values,100,cmap='jet')
    plt.pause(.001)
           

im=ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df.ws.loc[df.scan==scan_n].values,100,cmap='jet')
fig.colorbar(im)


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im=ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),df_median.ws.loc[df.scan==scan_n].values,np.linspace(-2,13,100),cmap='jet')
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im=ax.contourf(r_g*np.cos(p_g),r_g*np.sin(p_g),ws3_w_df.ws.loc[ws3_w_df.scan==scan_n].values,100,cmap='jet')
fig.colorbar(im)


# In[]

############# Re noise
#n = 0
#for dir_mean in Dir:
#    for u_mean in utot:
#        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
#            vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            if vlos0_file_name in onlyfiles:
#                noise0 = np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.5,azim_frac=.3),r_0_g.shape)
#                noise0 = noise0+np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.7,azim_frac=.6),r_0_g.shape)
#                noise0 = noise0+np.reshape(perlin_noise(r_0_g,phi_0_g,period=512,rad_lim=.9,azim_frac=.9),r_0_g.shape)
#                #normalize noise
#                #isolate areas without noise
#                ind_no_noise = (noise0 == 0)
#                a = np.max(noise0)
#                c = np.min(noise0)
#                if (a-c) > 0:
#                    b = 1
#                    d = -1   
#                    m = (b - d) / (a - c)
#                    noise0 = (m * (noise0 - c)) + d
#                noise0[ind_no_noise] = 0.0
#                noise0 = noise0*35.0
#                vlos0 = np.reshape(np.fromfile(vlos0_file_name, dtype=np.float32),r_0_g.shape)
#                vlos0_noise = vlos0 + noise0
#                (vlos0_noise.flatten()).astype(np.float32).tofile('noise0_'+vlos0_file_name)
#                n = n+1
#                print(n)


####################New data frame updated noise, rutine for noisy dataframe creation, should be a function
#with open('df_vlos_noise.pkl', 'rb') as reader:
#    df_vlos0_noise = pickle.load(reader)
#df_vlos0_noise['scan'] = df_vlos0_noise.groupby('azim').cumcount()
## reindex
#df_vlos0_noise.reset_index(inplace=True)
#scan = 0
#v_in = []
#m = np.arange(3,597,3)
#azim_unique = phi_0_g[:,0]*180/np.pi
#df_noise = pd.DataFrame(columns=df_vlos0_noise.columns[1:])
#aux_df_noise = np.zeros((len(azim_unique),len(df_vlos0_noise.columns[1:])))
#for dir_mean in Dir:
#    for u_mean in utot:
#        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
#            vlos0_file_name = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
#            if vlos0_file_name in onlyfiles:
#                n = 1
#                aux_df_noise[:,0] = azim_unique
#                v_in.append(vlos0_file_name)
#                ind = df_vlos0_noise_med.scan == scan
#                dataws = np.reshape(np.fromfile('noise0_'+vlos0_file_name, dtype=np.float32),r_0_g.shape)
#                dataCNR = df_vlos0_noise_med.CNR.loc[ind].values
#                datar = df_vlos0_noise_med.range_gate.loc[ind].values
#                for i in range(dataCNR.shape[1]):
#                    aux_df_noise[:,n:n+3] = np.c_[datar[:,i],dataws[:,i],dataCNR[:,i]]
#                    n = n+3
#                df_noise = pd.concat([df_noise, pd.DataFrame(data=aux_df_noise,
#                           index = df_vlos0_noise_med.index[ind],columns = df_vlos0_noise.columns[1:])])    
#                scan+=1
#                print(scan)
#df_noise['scan'] = df_noise.groupby('azim').cumcount()

################## saving the synthetic dataframe

#with open('df_syn_fin.pkl', 'wb') as writer:
#    pickle.dump(df_noise,writer) 
