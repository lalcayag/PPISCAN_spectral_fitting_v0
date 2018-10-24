# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:31:30 2018

@author: lalc

"""
import numpy as np
import pandas as pd
import pickle
import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize, abspath
from os import listdir
import sys
import tkinter as tkint

import tkinter.filedialog

import ppiscanprocess.filtering as filt
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectralconstruction as spec

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# In[Data loading]
# To do: it is necessary to link the mask file with the source file
root = tkint.Tk()
file_df = tkint.filedialog.askopenfilenames(parent=root,title='Choose a data file')
root.destroy()
root = tkint.Tk()
file_mask = tkint.filedialog.askopenfilenames(parent=root,title='Choose a mask file')
root.destroy()

# In[]

iden_lab = np.array(['start_id','stop_id','start_time','stop_time','azim','elev'])
labels = iden_lab
vel_lab = np.array(['range_gate','ws','CNR','Sb'])
for i in np.arange(198):
    labels = np.concatenate((labels,vel_lab))
sirocco_loc = np.array([6322832.3,0])
vara_loc = np.array([6327082.4,0])
d = vara_loc-sirocco_loc

# In[]

DF = []

for f,m in zip(file_df,file_mask):
    with open(m, 'rb') as reader:
        mask = pickle.load(reader)        
    df = pd.read_csv(f,sep=";", header=None) 
    df.columns = labels      
    df['scan'] = df.groupby('azim').cumcount()
  
    mask_CNR = (df.CNR>-24) & (df.CNR<-8)
    mask_CNR.columns =  mask.columns
    mask.mask(mask_CNR,other=False,inplace=True)
    df.ws=df.ws.mask(mask)
    DF.append(df)

       
phi0w = DF[0].azim.unique()
phi1w = DF[0].azim.unique()
r0w = np.array(DF[0].iloc[(DF[0].azim==min(phi0w)).nonzero()[0][0]].range_gate)
r1w = np.array(DF[0].iloc[(DF[0].azim==min(phi1w)).nonzero()[0][0]].range_gate)

r_vaw, phi_vaw = np.meshgrid(r0w, np.radians(phi0w)) # meshgrid
r_siw, phi_siw = np.meshgrid(r1w, np.radians(phi1w)) # meshgrid

treew,triw,wvaw,neighvaw,indexvaw,wsiw,neighsiw,indexsiw = wr.grid_over2((r_vaw,
                                                   phi_vaw),(r_siw, phi_siw),d)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(10000,13000):
    ax.cla()
    plt.title('Scan num. %i' %scan_n)
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),DF[1].ws.loc[DF[1].scan==scan_n].values,100,cmap='rainbow')
    plt.pause(.01)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(0,98):
    ax.cla()
    plt.title('Scan num. %i' %scan_n)
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df.ws.loc[df.scan==scan_n].values,100,cmap='rainbow')
    plt.pause(.01)
    
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)

for scan_n in range(0,98):
    ax.cla()
    plt.title('Scan num. %i' %scan_n)
    ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),dfp.ws.loc[dfp.scan==scan_n].values,100,cmap='rainbow')
    plt.pause(.01)
               
# In[Histograms]
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Alef']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

df_raw = pd.read_csv(file_df[0],sep=";", header=None)
df_raw.columns = labels      
df_raw['scan'] = df_raw.groupby('azim').cumcount()
df_clust2 = df_raw.copy()
df_clust2.ws=df_clust2.ws.mask(mask)   

ws_raw = df_raw.ws.values[~((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]

ws_clust2 = df_clust2.ws.values[~((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]

ws_raw_g = df_raw.ws.values[((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]

ws_clust2_g = df_clust2.ws.values[((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]

ws_median = ws_median[~np.isnan(ws_median)]
ws_clust2 = ws_clust2[~np.isnan(ws_clust2)]

ws_median_g = ws_median_g[~np.isnan(ws_median_g)]
ws_clust2_g = ws_clust2_g[~np.isnan(ws_clust2_g)]

r = df_raw.range_gate.values
a = (np.ones((r.shape[1],1))*df_raw.azim.values.flatten()).transpose()
r = r.flatten().astype(int)
a = a.flatten().astype(int)
n = np.max(df_raw.scan.values)
lim_m = [4,6,8]
ws_median = np.zeros((len(lim_m),len(ws_raw)))
ws_median_g = np.zeros((len(lim_m),len(ws_raw_g)))
recovery_median = np.zeros((len(lim_m),45*198))

for i,l in enumerate(lim_m[1:],1):
    df_median=filt.data_filt_median(df_raw,lim_m=l,lim_g=20)
    mask_median = np.isnan(df_median.ws).values
    mask_median = (~mask_median.flatten()).astype(int)
    ws_median[i,:] = df_median.ws.values[~((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
    ws_median_g[i,:] = df_median.ws.values[((df_raw.CNR>-24)&(df_raw.CNR<-8)).values]
    df_median = None
    df_s_median = pd.DataFrame({'r': r, 'a': a, 'm': mask_median})
    recovery_median[i,:] = df_s_median.groupby(['a', 'r'])['m'].agg('sum').values/n
    df_s_median = None

den=False

i = 2

h_raw,bine,_ = plt.hist(ws_raw.flatten(),bins=30,alpha=0.5,density=den)
h_med,_,_ = plt.hist(ws_median[i,:].flatten(),bins=bine,alpha=0.5,density=den)
h_clust,_,_ = plt.hist(ws_clust2.flatten(),bins=bine,alpha=0.5,density=den)

h_raw_g,bine_g,_ = plt.hist(ws_raw_g.flatten(),bins=30,alpha=0.5,density=den)
h_med_g,_,_ = plt.hist(ws_median_g[i,:].flatten(),bins=bine_g,alpha=0.5,density=den)
h_clust_g,_,_ = plt.hist(ws_clust2_g.flatten(),bins=bine_g,alpha=0.5,density=den)

bine=np.linspace(-5,5,50)
plt.figure()
plt.hist(((ws_median-np.mean(ws_median))/np.std(ws_median)).flatten(),bins=bine,alpha=0.5,density=True,histtype='step',color='blue',lw=2)
plt.hist(((ws_clust2-np.mean(ws_clust2))/np.std(ws_clust2)).flatten(),bins=bine,alpha=0.5,density=True,histtype='step',color='red',lw=2)
plt.plot(bine, np.exp(-bine**2/2)/np.sqrt(2*np.pi),'--k',lw=2)
plt.yscale('log')
plt.ylim(10**-3,1)
plt.xlabel(r'$V_{LOS}/\sigma$')
plt.ylabel('Prob. Density')

# In[]

phi1w = df_raw.azim.unique()
r1w = np.array(df_raw.iloc[(df_raw.azim==min(phi1w)).nonzero()[0][0]].range_gate)

r_siw, phi_siw = np.meshgrid(r1w, np.radians(phi1w)) # meshgrid

mask_median = np.isnan(df_median.ws).values


mask_median = (~mask_median.flatten()).astype(int)

df_s_median = pd.DataFrame({'r': r, 'a': a, 'm': mask_median})
mask_median = []

f, ax3 = plt.subplot(2, 2, 3)
im = ax3.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),np.reshape(recovery_median[2,:],phi_siw.shape),levels=np.linspace(.7,1,100),cmap='jet')
f.colorbar(im)
ax3.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax3.set_xlabel(r'North-South [m]', fontsize=12, weight='bold') 
    
r = df_raw.range_gate.values
a = (np.ones((r.shape[1],1))*df_raw.azim.values.flatten()).transpose()
r_clust = r[:mask.shape[0],:].astype(int)
a_clust = a[:mask.shape[0],:].astype(int)
mask = ((~mask.values)).astype(int)


df_s_clust = pd.DataFrame({'r': r_clust.flatten(), 'a': a_clust.flatten(), 'mask': mask.flatten()})
recovery_clust=np.reshape(df_s_clust.groupby(['a', 'r'])['mask'].agg('sum').values,r_siw.shape)
recovery_clust=recovery_clust/n
plt.figure()
plt.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),recovery_clust,levels=np.linspace(.6,1,100),cmap='jet')
plt.colorbar()

df_ws_clust = pd.DataFrame({'r': r_clust.flatten(), 'a': a_clust.flatten(), 'm': df_clust2.ws.values[:mask.shape[0],:].flatten()})
recovery_clust_std=np.reshape(df_ws_clust.groupby(['a', 'r'])['m'].agg('std').values,r_siw.shape)

df_ws_median = pd.DataFrame({'r': r, 'a': a, 'm': df_median.ws.values.flatten()})
recovery_median_std = df_ws_median.groupby(['a', 'r'])['m'].agg('std').values

# In[]
plt.figure()
i = 1

h_raw,bine,_ = plt.hist(ws_raw.flatten(),bins=30,alpha=0.5,density=den)
h_med,_,_ = plt.hist(ws_median[i,:].flatten(),bins=bine,alpha=0.5,density=den)
h_clust,_,_ = plt.hist(ws_clust2.flatten(),bins=bine,alpha=0.5,density=den)

h_raw_g,bine_g,_ = plt.hist(ws_raw_g.flatten(),bins=30,alpha=0.5,density=den)
h_med_g,_,_ = plt.hist(ws_median_g[i,:].flatten(),bins=bine_g,alpha=0.5,density=den)
h_clust_g,_,_ = plt.hist(ws_clust2_g.flatten(),bins=bine_g,alpha=0.5,density=den)

f, axs = plt.subplots(2,2)
axs=axs.flatten()
ax1=axs[0];ax2=axs[1];ax3=axs[2];ax4=axs[3]

#plt.step(.5*(bine[1:]+bine[:-1]),h_raw,color='blue',lw=3,label=r'Raw data')
#ax1 = plt.subplot(2, 2, 1)
ax1.step(.5*(bine[1:]+bine[:-1]),h_med/h_raw,color='black',lw=3,label=r'Median filter')
ax1.step(.5*(bine[1:]+bine[:-1]),h_clust/h_raw,color='red',lw=3,label=r'Density based filter')
#plt.yscale('log')
ax1.set_xlabel(r'$V_{LOS}$')
ax1.set_ylabel(r'Data recovery fraction')
ax1.legend(loc=(.7,.7))
ax1.set_xlim(-30,30)



#plt.step(.5*(bine[1:]+bine[:-1]),h_raw,color='blue',lw=3,label=r'Raw data')
ax2.step(.5*(bine[1:]+bine[:-1]),h_med_g/h_raw_g,color='black',lw=3,label=r'Median filter')
ax2.step(.5*(bine[1:]+bine[:-1]),h_clust_g/h_raw_g,color='red',lw=3,label=r'Density based filter')
#plt.yscale('log')
ax2.set_xlabel(r'$V_{LOS}$')
ax2.set_ylabel(r'Data recovery fraction')
ax2.legend(loc=(.7,.7))
ax2.set_xlim(-35,30)


im1 = ax3.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),np.reshape(recovery_median[i,:],phi_siw.shape),levels=np.linspace(.7,1,100),cmap='jet')

#divider1 = make_axes_locatable(ax3)
#cax1 = divider1.append_axes("right", size="5%", pad=0.05)
#cbar1 = f.colorbar(im1, cax=cax1)
#cbar1.ax.set_ylabel("Data recovery fraction")
ax3.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax3.set_xlabel(r'North-South [m]', fontsize=12, weight='bold') 
    
#f, ax4 = plt.subplot(2, 2, 4)
im2 = ax4.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),np.reshape(recovery_clust,phi_siw.shape),levels=np.linspace(.7,1,100),cmap='jet')
divider2 = make_axes_locatable(ax4)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = f.colorbar(im2, cax=cax2)
cbar1.ax.set_ylabel("Data recovery fraction")
ax4.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
ax4.set_xlabel(r'North-South [m]', fontsize=12, weight='bold') 


ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=24,
        verticalalignment='top')
ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=24,
       verticalalignment='top')
ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes, fontsize=24,
        verticalalignment='top')
ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontsize=24,
        verticalalignment='top') 


# In[]
             
U = []
V = []
#dphi200_clust0 = []
#dphi200_clust1 = []

for scan_n in range(min(df_sirocco.scan.max(),df_vara.scan.max())):
    print(scan_n)
    tot_s = (8910-mask_s[df_sirocco.scan==scan_n].sum().sum())/8910
    tot_v = (8910-mask_v[df_vara.scan==scan_n].sum().sum())/8910
    if (tot_s>.3) & (tot_v>.3):
        print(scan_n)    
        Lidar_sir = (df_sirocco.ws.loc[df_sirocco.scan==scan_n],phi_siw,wsiw,neighsiw,indexsiw) 
        Lidar_var = (df_vara.ws.loc[df_vara.scan==scan_n],phi_vaw,wvaw,neighvaw,indexvaw)
        auxU, auxV= wind_field_rec(Lidar_var, Lidar_sir, treew, triw, d)
        U.append(auxU) 
        V.append(auxV)
    else:
        U.append([]) 
        V.append([])

with open('U_'+file_v[:14]+'.pkl', 'wb') as U_clust:
    pickle.dump(U, U_clust)
    
with open('V_'+file_v[:14]+'.pkl', 'wb') as V_clust:
    pickle.dump(V, V_clust) 

# In[]
    
Uint, Vint = data_interp_triang(U_c,V_c,triw.x,triw.y,45)

with open('U_int_'+file_v[:14]+'.pkl', 'wb') as U_200_int:
    pickle.dump(Uint, U_200_int)
    
with open('V_int_'+file_v[:14]+'.pkl', 'wb') as V_200_int:
    pickle.dump(Vint, V_200_int)
 
# In[]

with open('U_int_'+file_v[:14]+'.pkl', 'rb') as U_200_int:
   Uint = pickle.load(U_200_int)
    
with open('V_int_'+file_v[:14]+'.pkl', 'rb') as V_200_int:
   Vint = pickle.load(V_200_int)
# In[]
    
fig, ax = plt.subplots()
ax.set_aspect('equal')
# Enforce the margins, and enlarge them to give room for the vectors.
ax.use_sticky_edges = False
ax.margins(0.07)
        
for i in range(8000,10000):
    #triw.set_mask(masks2[i])
    #tri_r.set_mask(masks_r[i])
    ax.cla()
    ax.set_aspect('equal')
    # Enforce the margins, and enlarge them to give room for the vectors.
    ax.use_sticky_edges = False
    ax.margins(0.07)
    plt.title('Scan num. %i' %i)
    ax.triplot(triw, color='black',lw=.5)
    U_mean = avetriangles(np.c_[triw.x,triw.y], Uint[i], triw.triangles)
    V_mean = avetriangles(np.c_[triw.x,triw.y], Vint[i], triw.triangles)
    
    im=ax.tricontourf(triw,np.sqrt(Uint[i]**2+Vint[i]**2),levels=np.linspace(10,20,300),cmap='jet')
    #im=ax.tricontourf(triw,Uint[i]-U_mean,levels=np.linspace(-10,5,300),cmap='jet')

    
    Q = ax.quiver(3000.00,-830.00,V_mean,U_mean,pivot='middle', scale=75, units='width', color='k')
    circle = plt.Circle((3000.00,-830.00), 450, color='k', fill=False)
    ax.add_artist(circle)
    ax.set_ylabel(r'West-East [m]', fontsize=12, weight='bold')
    ax.set_xlabel(r'North-South [m]', fontsize=12, weight='bold') 
    
    if len(fig.axes) > 1: 
        # if so, then the last axes must be the colorbar.
        # we get its extent
        pts = fig.axes[-1].get_position().get_points()
        # and its label
        label = fig.axes[-1].get_ylabel()
        # and then remove the axes
        fig.axes[-1].remove()
        # then we draw a new axes a the extents of the old one
        divider = make_axes_locatable(ax)
        cax= divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.set_ylabel("Wind speed [m/s]")
        # unfortunately the aspect is different between the initial call to colorbar 
        #   without cax argument. Try to reset it (but still it's somehow different)
        #cbar.ax.set_aspect(20)
    else:
        fig.colorbar(im)
    plt.pause(.5)

# In[]
Su_u = np.zeros((15050,256))
Sv_v = np.zeros((15050,256))
Su_v = np.zeros((15050,256))
k_1 = np.zeros((15050,256))
k_2 = np.zeros((15050,256))

for scan in range(0,15050):
    print(scan)
    if len(Uint[scan])>0:
        Su,Sv,Suv,k1,k2=spatial_autocorr_fft(triw,Vint[scan],Uint[scan],transform = True,N_grid=256,interp='cubic')
        Su_u[scan,:] = sp.integrate.simps(Su,k2,axis=1)
        Sv_v[scan,:] = sp.integrate.simps(Sv,k2,axis=1)
        Su_v[scan,:] = sp.integrate.simps(Suv,k2,axis=1)
        k_1[scan,:] = k1
        k_1[scan,:] = k2


for scan in range(0,15050): 
    if len(Uint[scan])>0:       
        k1,k2 = wavenumber(triw,Vint[scan],Uint[scan],N_grid=256) 
        k_1[scan,:] = k1
        k_1[scan,:] = k2
        C = 1/(4*max(k1)*max(k2))
        print(scan,C**2)
        Su_v[scan,:] = Su_v[scan,:]*C**2 
        
with open('S_minE_k_1_2.pkl', 'wb') as V_t:
     pickle.dump((Su_u,Sv_v,Su_v,k_1,k_2),V_t)    

fig1, ax1 = plt.subplots()
ax1.set_xscale('log')
ax1.set_yscale('log')

for i in range(0,len(Su_u)):
    ax1.scatter(S1[3],S1[3]*Su_u[i,:],color='black')
    
# In[]

X = RobustScaler(quantile_range=(25, 75)).fit_transform(Su_u)
tree_X = KDTree(X)
# first k distance estimation
d,i = tree_X.query(tree_X.data,k = 5)  
#k-nearest distance
d=d[:,-1]
# log transformation to level-up values and easier identification of "knees"
# d is an array with k-distances sorted in increasing value.
d = np.log(np.sort(d))
# x axis (point label)
l = np.arange(0,len(d))  
# Down sampling to speed up calculations
d_resample = np.array(d[::int(len(d)/400)])
print(len(d_resample))
# same with point lables
l_resample = l[::int(len(d)/400)]
# Cubic spline of resampled k-distances, lower memory usage and higher calculation speed.
#spl = UnivariateSpline(l_resample, d_resample,s=0.5)
std=.001*np.ones_like(d_resample)
# Changes in slope in the sorted, log transformed, k-distance graph
t = np.arange(l_resample.shape[0])    
fx = UnivariateSpline(t, l_resample/(-l_resample[0]+l_resample[-1]), k=4, w=1/std)
fy = UnivariateSpline(t, d_resample/(-d_resample[0]+d_resample[-1]), k=4, w=1/std)
x_1prime = fx.derivative(1)(t)
x_2prime = fx.derivative(2)(t)
y_1prime = fy.derivative(1)(t)
y_2prime = fy.derivative(2)(t) 
kappa = (x_1prime* y_2prime - y_1prime* x_2prime) / np.power(x_1prime**2 + y_1prime**2, 1.5)
# location of knee (first point with k-distance above 1 std of k-distance mean)
#ind_kappa = np.argsort(np.abs(kappa-(np.mean(kappa)+1*np.std(kappa)))) 
ind_kappa, _ = find_peaks(kappa,prominence=1) 
# Just after half of the graph
ind_kappa = ind_kappa[ind_kappa>int(.3*len(kappa))]
# The first knee...
l1 = l_resample[ind_kappa][0]
# the corresponding eps distance
eps0 = np.exp(d[l1])   
plt.plot(l_resample,d_resample) 

clf = DBSCAN(eps=eps0, min_samples=5)
clf.fit(X)
# Cluster-core samples and boundary samples identification
core_samples_mask = np.zeros_like(clf.labels_, dtype=bool)
core_samples_mask[clf.core_sample_indices_] = True
# Array with labels identifying the different clusters in the data
labels = clf.labels_
# Number of identified clusters (excluding noise)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 