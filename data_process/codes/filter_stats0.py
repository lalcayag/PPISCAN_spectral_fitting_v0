# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:31:30 2018

@author: lalc

"""
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import os
from subprocess import Popen, PIPE
from os.path import isfile, join, getsize, abspath
from os import listdir
import sys
import tkinter as tkint

import tkinter.filedialog

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


import ppiscanprocess.filtering as filt
import ppiscanprocess.windfieldrec as wr
import ppiscanprocess.spectraconstruction as spec

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from os import listdir
from os.path import isfile, join

import matplotlib.ticker as ticker

from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import RobustScaler

import importlib

import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

# In[For figures]

import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Wedge, Circle

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

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

def fm(x, pos=None):
    return r'${}$'.format('{:.2f}'.format(x).split('f')[0])

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

# In[Load data form databases]




# In[First figures]
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
    
################ Figure for CNR comb #############

#with open('Z_9025_9030.pkl', 'wb') as sim:
#     pickle.dump((Z,kernel),sim)
#     
def mycmap(c,alpha):
    import matplotlib
    c = []
    for i in range(alpha.shape[0]):
            c.append(matplotlib.colors.to_rgba('grey', alpha=alpha[i]))
    return c    
     
    
ind_can = (df.scan > 9000) & (df.scan < 9030)

df_sel = df.loc[ind_can]

df_sel_mask = df_sel.copy()
df_sel_vlos = df_sel.copy()
df_sel_clust = df_sel.copy()
mask_CNR = (df_sel.CNR>-27)
mask_vlos = (df_sel.ws>-21) & (df_sel.ws<0)
mask_CNR.columns =  df_sel_mask.ws.columns
df_sel_mask.ws=df_sel_mask.ws.mask(~mask_CNR)
df_sel_vlos.ws=df_sel_vlos.ws.mask(~mask_vlos)
df_sel_clust.ws=df_sel_vlos.ws.mask(mask)

scan_n = 9020

phi0 = df_sel.azim.unique()-90
r0 = df_sel.iloc[0].range_gate.values
r_0, phi_0 = np.meshgrid(r0, np.radians(phi0)) # meshgrid

ind_can_s = df_sel.scan==scan_n

shape = df_sel.loc[ind_can_s].ws.values.shape
pos = np.c_[df_sel.loc[ind_can_s].ws.values.flatten(),
            df_sel.loc[ind_can_s].CNR.values.flatten()]
kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(pos)
Z = kernel.score_samples(pos)
Z = np.exp(Z)

f=16
fig, ax = plt.subplots(2,2,figsize=(8,8))
fig.set_size_inches(16,16)
ax[0,0].set_aspect('equal')
ax[0,0].use_sticky_edges = False
ax[0,0].margins(0.1)
im = ax[0,0].contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_sel.ws.loc[df_sel.scan==scan_n].values,cmap='jet')
ax[0,0].set_xlabel('Easting m', fontsize=f)
ax[0,0].set_ylabel('Northing m', fontsize=f)
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax[0,0].tick_params(labelsize=14)
cbar.ax.set_ylabel("$V_{LOS}$ m/s", fontsize=f)
ax[0,0].text(0.05, 0.95, '(a)', transform=ax[0,0].transAxes, fontsize=24,
        verticalalignment='top')

ax[0,1].set_aspect('equal')
ax[0,1].use_sticky_edges = False
ax[0,1].margins(0.1)
im = ax[0,1].contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_sel_mask.ws.loc[df_sel_mask.scan==scan_n].values,cmap='jet')
ax[0,1].set_xlabel('Easting m', fontsize=f)
ax[0,1].set_ylabel('Northing m', fontsize=f)
divider = make_axes_locatable(ax[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax[0,1].tick_params(labelsize=14)
cbar.ax.set_ylabel("$V_{LOS}$ m/s", fontsize=f)
ax[0,1].text(0.05, 0.95, '(b)', transform=ax[0,1].transAxes, fontsize=24,
        verticalalignment='top')

ax[1,0].set_aspect('equal')
ax[1,0].use_sticky_edges = False
ax[1,0].margins(0.1)
im = ax[1,0].contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_sel_vlos.ws.loc[df_sel_vlos.scan==scan_n].values,cmap='jet')
ax[1,0].set_xlabel('Easting m', fontsize=f)
ax[1,0].set_ylabel('Northing m', fontsize=f)
divider = make_axes_locatable(ax[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax[1,0].tick_params(labelsize=14)
cbar.ax.set_ylabel("$V_{LOS}$ m/s", fontsize=f)
ax[1,0].text(0.05, 0.95, '(c)', transform=ax[1,0].transAxes, fontsize=24,
        verticalalignment='top')

#ax[1,1].set_aspect('equal')
#ax[1,1].use_sticky_edges = False
#ax[1,1].margins(0.1)
#im = ax[1,1].contourf(r_0*np.cos(phi_0),r_0*np.sin(phi_0),df_sel_clust.ws.loc[df_sel_clust.scan==scan_n].values,cmap='jet')
#ax[1,1].set_xlabel('Easting m', fontsize=f)
#ax[1,1].set_ylabel('Northing m', fontsize=f)
#divider = make_axes_locatable(ax[1,1])
#cax = divider.append_axes("right", size="5%", pad=0.05)
#cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
#cbar.ax.tick_params(labelsize=f)
#ax[1,1].tick_params(labelsize=14)
#cbar.ax.set_ylabel("$V_{LOS}$ m/s", fontsize=f)
#ax[1,1].text(0.05, 0.95, '(c)', transform=ax[1,1].transAxes, fontsize=24,
#        verticalalignment='top')
#fig.tight_layout()

ax[1,1].set_aspect('equal')
ax[1,1].use_sticky_edges = False
ax[1,1].margins(0.1)
im = ax[1,1].scatter(pos[:,0],pos[:,1], s=20, c = np.log(Z), cmap='jet')  #df.loc[ind_can].plot.scatter(x='ws', y='CNR', c='grey', s=20, edgecolor='k',cmap = 'jet')  
ax[1,1].set_xlabel('$V_{LOS}$ m/s',fontsize=16) 
ax[1,1].set_ylabel('CNR dB',fontsize=16)  
ax[1,1].plot([-40,40],[-27,-27],'--r',lw=2)
ax[1,1].plot([-21,-21],[-40,-0],'--k',lw=2)
ax[1,1].plot([0,0],[-40,-0],'--k',lw=2)
ax[1,1].set_xlim([-35,35])
ax[1,1].set_ylim([-40,0])
ax[1,1].tick_params(axis='both', which='major', labelsize=14)
divider = make_axes_locatable(ax[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, cax=cax, format=ticker.FuncFormatter(fm))
cbar.ax.tick_params(labelsize=f)
ax[1,1].tick_params(labelsize=14)
cbar.ax.set_ylabel("log(KDE)", fontsize=f)
ax[1,1].text(0.05, 0.95, '(d)', transform=ax[1,1].transAxes, fontsize=24,
        verticalalignment='top')
fig.tight_layout()

fig.savefig(r'C:\Users\lalc\Documents\Old Documents folder\Publications\Filtering using a clustering algorithm\Figures\CNR_comb')


########################

ind_can_s = df_sel.scan==scan_n

shape = df_sel.loc[ind_can_s].ws.values.shape
pos3d = np.c_[df_sel.loc[ind_can_s].ws.values.flatten(), df_sel.loc[ind_can_s].range_gate.values.flatten(),
            df_sel.loc[ind_can_s].CNR.values.flatten()]
X = RobustScaler(quantile_range=(25, 75)).fit_transform(pos3d) 
kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
Z = kernel.score_samples(X)
size = (np.exp(Z)-np.min(np.exp(Z)))/(np.max(np.exp(Z))-np.min(np.exp(Z)))

import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
fig.set_size_inches(20,8)
xx,yy,zz = np.meshgrid(np.array([-35,35]),np.array([0,7000]),np.array([-40,-27,0]))

ax = fig.add_subplot(1,2,1,projection='3d')
ax.voxels(xx, yy, zz, zz[:-1,:-1,:-1] < -27, facecolors='grey', alpha=.1, edgecolor = 'k')
ax.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], s=10, c= np.exp(Z),cmap = 'jet',zorder = 1)
ax.set_xlabel('$V_{LOS}$ m/s',fontsize=16) 
ax.set_zlabel('CNR dB',fontsize=16)  
ax.set_ylabel('Range gate m',fontsize=16)  
ax.set_xlim([-35,35])
ax.set_ylim([7000,0])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(10, 10, 250, '(a)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

xx,yy,zz = np.meshgrid(np.array([-35,-21,0,35]),np.array([0,7000]),np.array([-40,0]))
cub = (xx>=0) | (xx<-21)

ax = fig.add_subplot(1,2,2,projection='3d')
ax.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], s=10, c= np.exp(Z),cmap = 'jet')
ax.voxels(xx, yy, zz, cub[:-1,:-1,:-1], facecolors='grey', alpha=.1, edgecolor = 'k')
ax.set_xlabel('$V_{LOS}$ m/s',fontsize=16) 
ax.set_zlabel('CNR dB',fontsize=16)  
ax.set_ylabel('Range gate m',fontsize=16)  
ax.set_xlim([-35,35])
ax.set_ylim([7000,0])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(10, 10, 250, '(b)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.view_init(30, -100)
ax.grid(False)

fig.tight_layout()


#ax.plot_surface(ws_plane,r_plane,CNR_plane, color = 'r', alpha = .5, zorder=1)
#face1 = mp3d.art3d.Poly3DCollection([top], alpha=0.5, linewidth=1)
#alpha = 0.5
#face1.set_facecolor((0, 0, 1, alpha))
#ax.add_collection3d(face1)

###########
#idxcnr = df.loc[ind_can].CNR.values.flatten() >-27
#idxws = (df.loc[ind_can].ws.values.flatten()>-23) & (df.loc[ind_can].ws.values.flatten()<0)
#idxr = (df.loc[ind_can].range_gate.values.flatten()>4000)
#idxwsr = idxws & ~idxcnr & idxr
#colors = np.array(['grey']*len(idxcnr))
#colors[~idxcnr] = 'red'
#colors[idxwsr] = 'blue'
#############
#
#ax.scatter(df.loc[ind_can].ws.values.flatten()[~idxcnr], df.loc[ind_can].range_gate.values.flatten()[~idxcnr],
#           df.loc[ind_can].CNR.values.flatten()[~idxcnr], c = 'red', alpha=0.2, s=20, edgecolor='k',zorder = 1)
#ax.scatter(df.loc[ind_can].ws.values.flatten()[idxcnr], df.loc[ind_can].range_gate.values.flatten()[idxcnr],
#           df.loc[ind_can].CNR.values.flatten()[idxcnr], c = 'grey',s=20, edgecolor='k',zorder = 10)   

#ax.scatter(df.loc[ind_can].ws.values.flatten(), df.loc[ind_can].range_gate.values.flatten(),
#           df.loc[ind_can].CNR.values.flatten(), c = colors, s=20, edgecolor='k')

############
#ax.scatter(df.loc[ind_can].ws.values.flatten(), df.loc[ind_can].range_gate.values.flatten(),
#           df.loc[ind_can].CNR.values.flatten(), c = 'grey', s=20, edgecolor='k',alpha=.2)
############### 


df_clust = df.copy()
df_clust.ws=df_clust.ws.mask(mask)


ind_can = (df.scan > 9000) & (df.scan < 9030)
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

idxws = np.isnan(df_clust.loc[ind_can].ws.values.flatten())
colors = np.array(['grey']*len(idxws))
colors[idxws] = 'red'


ax.scatter(df.loc[ind_can].ws.values.flatten()[~idxws], df.loc[ind_can].range_gate.values.flatten()[~idxws],
           df.loc[ind_can].CNR.values.flatten()[~idxws], c = colors[~idxws], s=20, edgecolor='k')

ax.set_xlabel('$V_{LOS}$',fontsize=16) 
ax.set_zlabel('$CNR$',fontsize=16)  
ax.set_ylabel('$Range\:gate$',fontsize=16)  
ax.set_xlim([-35,35])
ax.set_ylim([7000,0])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(10, 10, 250, '(b)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
ax.grid(False)



df_median=fl.data_filt_median(df, lim_m= 3 , n = 4, m = 3) 

ind_can = (df.scan > 9000) & (df.scan < 9030)
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')


idxws = np.isnan(df_median.loc[ind_can].ws.values.flatten())
colors = np.array(['grey']*len(idxws))
colors[idxws] = 'red'
#
#ax.scatter(df.loc[ind_can].ws.values.flatten()[~idxcnr], df.loc[ind_can].range_gate.values.flatten()[~idxcnr],
#           df.loc[ind_can].CNR.values.flatten()[~idxcnr], c = 'red', alpha=0.2, s=20, edgecolor='k',zorder = 1)
#ax.scatter(df.loc[ind_can].ws.values.flatten()[idxcnr], df.loc[ind_can].range_gate.values.flatten()[idxcnr],
#           df.loc[ind_can].CNR.values.flatten()[idxcnr], c = 'grey',s=20, edgecolor='k',zorder = 10)   

ax.scatter(df.loc[ind_can].ws.values.flatten(), df.loc[ind_can].range_gate.values.flatten(),
           df.loc[ind_can].CNR.values.flatten(), c = colors, s=20, edgecolor='k')

ax.set_xlabel('$V_{LOS}$',fontsize=16) 
ax.set_zlabel('$CNR$',fontsize=16)  
ax.set_ylabel('$Range\:gate$',fontsize=16)  
ax.set_xlim([-35,35])
ax.set_ylim([7000,0])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.text(10, 10, 250, '(b)', transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
ax.grid(False)

##################################################
df_median0 = fl.data_filt_median(df, lim_m= 3 , n = 4, m = 3) 
df_median1 = fl.data_filt_median(df, lim_m= 5 , n = 4, m = 5) 

idxws = np.isnan(df_median0.loc[ind_can].ws.values.flatten())
colors = np.array(['grey']*len(idxws))
colors[idxws] = 'red'
plt.figure()
plt.scatter(df.loc[ind_can].ws.values.flatten()[~idxws], df.loc[ind_can].CNR.values.flatten()[~idxws], c = colors[~idxws], s=20, edgecolor='k', label = 'Accepted obs.')
plt.scatter(df.loc[ind_can].ws.values.flatten()[idxws], df.loc[ind_can].CNR.values.flatten()[idxws], c = colors[idxws], s=20, edgecolor='k',alpha=.2, label = 'Filtered obs.')
plt.xlabel('$V_{LOS}$')
plt.ylabel('$CNR$')
plt.legend()

idxws = np.isnan(df_median1.loc[ind_can].ws.values.flatten())
colors = np.array(['grey']*len(idxws))
colors[idxws] = 'red'
plt.figure()
plt.scatter(df.loc[ind_can].ws.values.flatten()[~idxws], df.loc[ind_can].CNR.values.flatten()[~idxws], c = colors[~idxws], s=20, edgecolor='k', label = 'Accepted obs.')
plt.scatter(df.loc[ind_can].ws.values.flatten()[idxws], df.loc[ind_can].CNR.values.flatten()[idxws], c = colors[idxws], s=20, edgecolor='k',alpha=.2, label = 'Filtered obs.')
plt.xlabel('$V_{LOS}$')
plt.ylabel('$CNR$')
plt.legend()

idxws = np.isnan(df_clust.loc[ind_can].ws.values.flatten())
colors = np.array(['grey']*len(idxws))
colors[idxws] = 'red'
plt.figure()
plt.scatter(df.loc[ind_can].ws.values.flatten()[~idxws], df.loc[ind_can].CNR.values.flatten()[~idxws], c = colors[~idxws], s=20, edgecolor='k', label = 'Accepted obs.')
plt.scatter(df.loc[ind_can].ws.values.flatten()[idxws], df.loc[ind_can].CNR.values.flatten()[idxws], c = colors[idxws], s=20, edgecolor='k',alpha=.2, label = 'Filtered obs.')
plt.xlabel('$V_{LOS}$')
plt.ylabel('$CNR$')
plt.legend()

idxws = np.isnan(df_cluster.loc[ind_can].ws.values.flatten())
colors = np.array(['grey']*len(idxws))
colors[idxws] = 'red'
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(df.loc[ind_can].ws.values.flatten()[~idxws],df.loc[ind_can].range_gate.values.flatten()[~idxws], df.loc[ind_can].Sb.values.flatten()[~idxws], c = colors[~idxws], s=20, edgecolor='k')
ax.scatter(df.loc[ind_can].ws.values.flatten()[idxws],df.loc[ind_can].range_gate.values.flatten()[idxws], df.loc[ind_can].Sb.values.flatten()[idxws], c = colors[idxws], s=20, edgecolor='k',alpha=.2)

##################################################

scan_n = 9015

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df_median.ws.loc[df_median.scan==scan_n].values,100,cmap='rainbow')
isnan = np.isnan(df_median.ws.loc[df_median.scan==scan_n].values)
ax.scatter((r_vaw*np.cos(phi_vaw))[isnan],(r_vaw*np.sin(phi_vaw))[isnan],c='k',alpha=.2) 
fig.colorbar(im)

isnan = np.isnan(df_clust.ws.loc[df_clust.scan==scan_n].values)


fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df.ws.loc[df.scan==scan_n].values,100,cmap='rainbow')
ax.scatter((r_vaw*np.cos(phi_vaw))[isnan],(r_vaw*np.sin(phi_vaw))[isnan],c='k',alpha=.2) 
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df_clust.ws.loc[df_clust.scan==scan_n].values,100,cmap='rainbow')
ax.scatter((r_vaw*np.cos(phi_vaw))[isnan],(r_vaw*np.sin(phi_vaw))[isnan],c='k',alpha=.2) 
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df_clust.CNR.loc[df_clust.scan==scan_n].values,100,cmap='rainbow')
ax.scatter((r_vaw*np.cos(phi_vaw))[isnan],(r_vaw*np.sin(phi_vaw))[isnan],c='k',alpha=.2) 
fig.colorbar(im)


cnr = df_clust.CNR.loc[df_clust.scan==scan_n].values
cnr[cnr<-27] = np.nan

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),cnr,100,cmap='rainbow')
ax.scatter((r_vaw*np.cos(phi_vaw))[isnan],(r_vaw*np.sin(phi_vaw))[isnan],c='k',alpha=.2) 
fig.colorbar(im)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
im = ax.contourf(r_vaw*np.cos(phi_vaw),r_vaw*np.sin(phi_vaw),df_clust.Sb.loc[df_clust.scan==scan_n].values,100,cmap='rainbow')
ax.scatter((r_vaw*np.cos(phi_vaw))[isnan],(r_vaw*np.sin(phi_vaw))[isnan],c='k',alpha=.2)  
fig.colorbar(im)

plt.figure()
plt.hist((df_clust.Sb.loc[df_clust.scan==scan_n].values)[isnan],bins=30, alpha=.2, density = True)
plt.hist((df_clust.Sb.loc[df_clust.scan==scan_n].values)[~isnan],bins=30, alpha=.2, density = True)

plt.figure()
plt.hist((df_clust.CNR.loc[df_clust.scan==scan_n].values)[isnan],bins=30, alpha=.2, density = True)
plt.hist((df_clust.CNR.loc[df_clust.scan==scan_n].values)[~isnan],bins=30, alpha=.2, density = True)



##################################################
       
phi0w = df.azim.unique()
phi1w = df.azim.unique()
r0w = np.array(df.iloc[(df.azim==min(phi0w)).nonzero()[0][0]].range_gate)
r1w = np.array(df.iloc[(df.azim==min(phi1w)).nonzero()[0][0]].range_gate)

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
df_median=filt.data_filt_median(df_raw,lim_m=6,lim_g=100) 

ws_raw = df.ws.values[~((df.CNR>-24)&(df.CNR<-8)).values]
ws_clust = df_clust.ws.values[~((df.CNR>-24)&(df.CNR<-8)).values]
ws_median0 = df_median0.ws.values[~((df.CNR>-24)&(df.CNR<-8)).values]
ws_median1 = df_median1.ws.values[~((df.CNR>-24)&(df.CNR<-8)).values]

ws_raw_g = df.ws.values[((df.CNR>-24)&(df.CNR<-8)).values]
ws_clust_g = df_clust.ws.values[((df.CNR>-24)&(df.CNR<-8)).values]
ws_median_g0 = df_median0.ws.values[((df.CNR>-24)&(df.CNR<-8)).values]
ws_median_g1 = df_median1.ws.values[((df.CNR>-24)&(df.CNR<-8)).values]

#ws_median = ws_median[~np.isnan(ws_median)]
#ws_clust2 = ws_clust2[~np.isnan(ws_clust2)]
#
#ws_median_g = ws_median_g[~np.isnan(ws_median_g)]
#ws_clust2_g = ws_clust2_g[~np.isnan(ws_clust2_g)]

r = df.range_gate.values
a = (np.ones((r.shape[1],1))*df.azim.values.flatten()).transpose()
r = r.flatten().astype(int)
a = a.flatten().astype(int)
n = np.max(df.scan.values)

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

# In[Recovery, spatial]
r = df.range_gate.values
a = (np.ones((r.shape[1],1))*df.azim.values.flatten()).transpose()
r_clust = r[:mask.shape[0],:].astype(int)
a_clust = a[:mask.shape[0],:].astype(int)
mask = ((~mask.values)).astype(int)

phi1w = df.azim.unique()
r1w = np.array(df.iloc[(df.azim==min(phi1w)).nonzero()[0][0]].range_gate)

r_siw, phi_siw = np.meshgrid(r1w, np.radians(phi1w)) # meshgrid

mask_median0 = np.isnan(df_median0.ws).values
mask_median1 = np.isnan(df_median1.ws).values

mask_median0 = (~mask_median0.flatten()).astype(int)
mask_median1 = (~mask_median1.flatten()).astype(int)

df_s_median0 = pd.DataFrame({'r': r, 'a': a, 'm': mask_median0})
mask_median0 = []
df_s_median1 = pd.DataFrame({'r': r, 'a': a, 'm': mask_median1})
mask_median1 = []

recovery_median0 = df_s_median0.groupby(['a', 'r'])['m'].agg('sum').values/n
df_s_median0 = []
recovery_median1 = df_s_median1.groupby(['a', 'r'])['m'].agg('sum').values/n
df_s_median1 = []
recovery_median0 = np.reshape(recovery_median0, r_siw.shape)
recovery_median1 = np.reshape(recovery_median0, r_siw.shape) 

df_s_clust = pd.DataFrame({'r': r_clust.flatten(), 'a': a_clust.flatten(), 'mask': mask.flatten()})
recovery_clust=np.reshape(df_s_clust.groupby(['a', 'r'])['mask'].agg('sum').values,r_siw.shape)
recovery_clust=recovery_clust/n
df_s_clust = []

fig, ax1 = plt.subplots()
im1 = ax1.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),
                   np.reshape(recovery_median1,phi_siw.shape),levels=np.linspace(.7,1,10),cmap='jet')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = fig.colorbar(im1, cax=cax1, format=ticker.FuncFormatter(fm))
cbar1.ax.tick_params(labelsize=16)
cbar1.ax.set_ylabel('$Data\:recovery\:fraction$', fontsize=16)
ax1.set_ylabel(r'$West-East\:[m]$', fontsize = 16, weight='bold')
ax1.set_xlabel(r'$North-South\:[m]$', fontsize = 16, weight='bold') 
ax1.tick_params(labelsize = 16)
ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=24,
        verticalalignment='top')
fig.tight_layout()
  
  
fig, ax2 = plt.subplots()
im2 = ax2.contourf(r_siw*np.cos(phi_siw),r_siw*np.sin(phi_siw),
                   np.reshape(recovery_clust,phi_siw.shape),levels=np.linspace(.7,1,10),cmap='jet')
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = fig.colorbar(im2, cax=cax2, format=ticker.FuncFormatter(fm))

cbar2.ax.set_ylabel('$Data\:recovery\:fraction$', fontsize=16)
cbar2.ax.tick_params(labelsize=16)
ax2.set_ylabel(r'$West-East\:[m]$', fontsize=16, weight='bold')
ax2.set_xlabel(r'$North-South\:[m]$', fontsize=16, weight='bold') 
ax2.tick_params(labelsize = 16)
ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=24,
        verticalalignment='top') 
fig.tight_layout()

#####################################

t_step = 3
ind=np.unique(df.scan.values)%t_step==0
times= np.unique(np.append(np.unique(df.scan.values)[ind],df.scan.values[-1]))

loc = (df.scan>=700) & (df.scan<703)
feat = ['ws','range_gate','CNR','azim','dvdr']
fl.data_filt_DBSCAN(df.loc[loc],feat,plot=True)


###############################


# In[Revocery, statistics]

plt.figure()

h_raw,bine,_ = plt.hist(ws_raw.flatten(),bins=60,alpha=0.5,density=den)
h_med0,_,_ = plt.hist(ws_median0.flatten(),bins=bine,alpha=0.5,density=den)
h_med1,_,_ = plt.hist(ws_median1.flatten(),bins=bine,alpha=0.5,density=den)
h_clust,_,_ = plt.hist(ws_clust.flatten(),bins=bine,alpha=0.5,density=den)

plt.figure()

h_raw_g,bine_g,_ = plt.hist(ws_raw_g.flatten(),bins=60,alpha=0.5,density=den)
h_med_g0,_,_ = plt.hist(ws_median_g0.flatten(),bins=bine_g,alpha=0.5,density=den)
h_med_g1,_,_ = plt.hist(ws_median_g1.flatten(),bins=bine_g,alpha=0.5,density=den)
h_clust_g,_,_ = plt.hist(ws_clust_g.flatten(),bins=bine_g,alpha=0.5,density=den)

indvalid_clust = ~np.isnan(ws_clust.flatten())
indquant_clust = ~((ws_clust.flatten()[indvalid_clust] > quantiles[0]) & (ws_clust.flatten()[indvalid_clust] < quantiles[1])) 
indvalid_med1 = ~np.isnan(ws_median1.flatten())
indquant_med1 = ~((ws_median1.flatten()[indvalid_med1] > quantiles[0]) & (ws_median1.flatten()[indvalid_med1] < quantiles[1]))
indquant_raw = ~((ws_raw.flatten() > quantiles[0]) & (ws_raw.flatten() < quantiles[1])) 

fraction_clust = np.sum(indquant_clust)/np.sum(indquant_raw)
fraction_med1 = np.sum(indquant_med1)/np.sum(indquant_raw)
fraction_noise = np.sum(indquant_raw)/len(indquant_raw)

fraction_recov = np.sum(indvalid_clust)/len(ws_raw_g.flatten())

#############################################################
    
fig, ax1 = plt.subplots()
ax1.step(.5*(bine[1:]+bine[:-1])[h_med0>0],h_med0[h_med0>0]/h_raw[h_med0>0],color='black', lw=3, label = '$(4,\:3,\:3)$')
ax1.step(.5*(bine[1:]+bine[:-1])[h_med0>0],h_med1[h_med0>0]/h_raw[h_med0>0],color='blue', lw=3, label = '$(4,\:5,\:5)$')
ax1.step(.5*(bine[1:]+bine[:-1])[h_med0>0],h_clust[h_med0>0]/h_raw[h_med0>0],color='red', lw=3, label = '$Clustering$')
quantiles = np.quantile(ws_raw_g.flatten(),q = [1-.997,.997])
ax1.fill_between([-30,quantiles[0]], [0,0], [1,1], facecolor='grey', alpha=.2)
ax1.fill_between([quantiles[1],35], [0,0], [1,1], facecolor='grey', alpha=.2)
tol = .3
x1 = (.5*(bine[1:]+bine[:-1])[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
y1 = (h_clust[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
y2 = (h_med1[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]>=quantiles[1]-tol]
ax1.fill_between(x1, y1, y2, step = 'pre', facecolor='grey', alpha=.8)
x1 = (.5*(bine[1:]+bine[:-1])[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
y1 = (h_clust[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
y2 = (h_med1[h_med0>0]/h_raw[h_med0>0])[.5*(bine[1:]+bine[:-1])[h_med0>0]<=quantiles[0]+tol]
ax1.fill_between(x1, y1, y2, step = 'pre', facecolor='grey', alpha=.8)
ax1.set_xlabel('$V_{LOS}$',fontsize=16)
ax1.set_ylabel('$Data\:recovery\:fraction$',fontsize=16)
ax1.legend(loc=(.62,.6),fontsize=16)
ax1.set_xlim(-30,35)
ax1.set_ylim(0,1)
ax1.tick_params(labelsize=16)
ax1.text(0.05, 0.95, '(b)', transform=ax1.transAxes, fontsize=24, verticalalignment='top')
fig.tight_layout()

h_raw_g_d,bine_g_d,_ = plt.hist(ws_raw_g.flatten(),bins=30,alpha=0.5,density=True)
h_raw_d,bine_d,_ = plt.hist(ws_raw.flatten()[ws_raw.flatten()<50],bins=30,alpha=0.5,density=True)

fig, ax2 = plt.subplots()
ax2.step(.5*(bine_g_d[1:]+bine_g_d[:-1]),h_raw_g_d,color='black',lw=3,label = r'$CNR\:\in:[-24,-8]$')
ax2.step(.5*(bine_d[1:]+bine_d[:-1]),h_raw_d,color='red',lw=3,label = r'$CNR\:\not\in:[-24,-8]$')
quantiles = np.quantile(ws_raw_g.flatten(),q = [1-.997,.997])
ax2.fill_between([.5*(bine_d[1:]+bine_d[:-1]).min(),quantiles[0]], [0,0], [.07,.07], facecolor='grey', alpha=.2)
ax2.fill_between([quantiles[1],.5*(bine_d[1:]+bine_d[:-1]).max()], [0,0], [.07,.07], facecolor='grey', alpha=.2)
ax2.set_ylim(0,.07)
ax2.set_xlim(-30,30)
ax2.set_xlabel('$V_{LOS}$',fontsize=16)
ax2.set_ylabel('$Probability\:density$',fontsize=16)
ax2.legend(loc=(.5,.7),fontsize=16)
ax2.tick_params(labelsize=16)
ax2.text(0.05, 0.95, '(a)', transform=ax2.transAxes, fontsize=24, verticalalignment='top')
fig.tight_layout()

###################################
fig, ax = plt.subplots()
ax.step(.5*(bine_g[1:]+bine_g[:-1])[h_raw_g>1],h_med_g0[h_raw_g>1]/h_raw_g[h_raw_g>1],color='black',lw=3,label = '$(4,\:3,\:3)$')
ax.step(.5*(bine_g[1:]+bine_g[:-1])[h_raw_g>1],h_med_g1[h_raw_g>1]/h_raw_g[h_raw_g>1],color='blue',lw=3,label = '$(4,\:5,\:5)$')
ax.step(.5*(bine_g[1:]+bine_g[:-1])[h_raw_g>1],h_clust_g[h_raw_g>1]/h_raw_g[h_raw_g>1],color='red',lw=3,label='$Clustering$')
ax.set_xlabel('$V_{LOS}$',fontsize=16)
ax.set_ylabel('$Data\:recovery\:fraction$',fontsize=16)
ax.legend(loc=(.62,.6),fontsize=16)
ax.set_xlim(-30,35)
ax.tick_params(labelsize=16)
ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=24, verticalalignment='top')
fig.tight_layout()
####################

# In[Synthetic lidars filtering test]
############################## 
##############################    
# In[Synthetic data loading]
root = tkint.Tk()
file_vlos = tkint.filedialog.askopenfilenames(parent=root,title='Choose vlos scans files')
root.destroy()

root = tkint.Tk()
file_vlos_noise = tkint.filedialog.askopenfilenames(parent=root,title='Choose contaminated vlos scans files')
root.destroy()

root = tkint.Tk()
file_in_path = tkint.filedialog.askdirectory(parent=root,title='Choose a sim. Input dir')
root.destroy()

###############
# Dataframe creation

r_0 = np.linspace(150,7000,198) # It is 105!!!!!!!!!!!!!!!!!!!!!!!!!
r_1 = np.linspace(150,7000,198)
phi_0 = np.linspace(256,344,45)*np.pi/180
phi_1 = np.linspace(196,284,45)*np.pi/180
r_0_g, phi_0_g = np.meshgrid(r_0,phi_0)
r_1_g, phi_1_g = np.meshgrid(r_1,phi_1)

ae = [0.025, 0.05, 0.075]
L = [125,250,500,750]
G = [2.5,3.5]
seed = np.arange(1,10)
ae,L,G,seed = np.meshgrid(ae,L,G,-seed)
utot = np.linspace(15,25,5)
Dir = np.linspace(90,270,5)*np.pi/180

onlyfiles = [f for f in listdir(file_in_path) if isfile(join(file_in_path, f))]

#####CNR#######
DT = int(600/45)
scanlist = np.arange(0,df.scan.max(),DT)
CNR = df.CNR.values
r = df.range_gate.values
scan = np.array(list(df.scan.values)*r.shape[1]).transpose()
azim = np.array(list(df.azim.values)*r.shape[1]).transpose()
r_unique = r[0,:]
azim_unique = df.azim.unique()
range_gate,azim = np.meshgrid(r_unique,azim_unique)
K_CNR = []
for r_i in r_unique:
    print(r_i)
    K_CNR_t = []
    for t in scanlist[:-1]:
        print(t)
        ind_r = r.flatten() == r_i
        ind_t = (scan.flatten() >=t) & (scan.flatten() < t+DT)
        ind_tot = (ind_r) & (ind_t)
        K_CNR_t.append(sp.stats.gaussian_kde(CNR.flatten()[ind_tot], bw_method='silverman'))
    K_CNR.append(K_CNR_t)
    
for r_i in r_unique:                                 # random scan 
    t = np.random.randint(scanlist[0],scanlist[-1])

    ind_t = (scan.flatten() >=t) & (scan.flatten() < t+DT)
    #ind_tot = (ind_r) & (ind_t)
    x = r.flatten()*np.cos(azim.flatten()*np.pi/180)
    y = r.flatten()*np.sin(azim.flatten()*np.pi/180)
    X = np.c_[CNR.flatten()[ind_t],x[ind_t],y[ind_t]]
    transformer = RobustScaler(quantile_range=(25, 75))
    transformer.fit(X)
    X = transformer.transform(X)  
    
    from sklearn.model_selection import GridSearchCV
    
    bandwidths = 10 ** np.linspace(-2, 0, 20)
    
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=2)
    grid.fit(X)
    
    kde = KernelDensity(kernel='gaussian', bandwidth=.021)
    kde.fit(X)
    samples = kde.sample(n_samples=45*198)
    samples = transformer.inverse_transform(samples)
    x_int = r_0_g*np.cos(phi_0_g)
    y_int = r_0_g*np.sin(phi_0_g)
    X_int = sp.interpolate.griddata(samples[:,1:],samples[:,0], (x_int.flatten(), y_int.flatten()), method='linear')

im=plt.contourf(x_int, y_int, np.reshape(X_int,r_0_g.shape),cmap='jet')
plt.colorbar(im)

with open('K_CNR.pkl', 'wb') as writer:
    pickle.dump(K_CNR,writer)   
#####Dataframes

iden_lab = np.array(['azim'])
labels = iden_lab

vel_lab = np.array(['range_gate','ws','CNR'])
for i in np.arange(198):
    labels = np.concatenate((labels,vel_lab))
df_vlos0 = pd.DataFrame(columns=labels)
df_vlos0_noise = pd.DataFrame(columns=labels)
df_vlos1 = pd.DataFrame(columns=labels)
df_vlos1_noise = pd.DataFrame(columns=labels)

r_unique0 = r_0_g[0,:]
azim_unique0 = phi_0_g[:,0]*180/np.pi
r_unique1 = r_1_g[0,:]
azim_unique1 = phi_1_g[:,0]*180/np.pi
aux_df = np.zeros((len(azim_unique),len(labels)))
aux_df_noise = np.zeros((len(azim_unique),len(labels)))

#############CNR
count=0
for dir_mean in Dir:
    for u_mean in utot:
        for ae_i,L_i,G_i,seed_i in zip(ae.flatten(),L.flatten(),G.flatten(),seed.flatten()):
            vlos0_file = 'vlos0'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            vlos1_file = 'vlos1'+str(u_mean)+str(int(dir_mean*180/np.pi))+str(L_i)+str(G_i)+str(ae_i)+str(seed_i)
            vlos0_noise_file = 'noise0_'+vlos0_file
            vlos1_noise_file = 'noise1_'+vlos1_file
            if vlos0_file in onlyfiles:                
                vlos0 = np.reshape(np.fromfile(vlos0_file, dtype=np.float32),r_0_g.shape)
                vlos0_noise = np.reshape(np.fromfile(vlos0_noise_file, dtype=np.float32),r_0_g.shape)
                n = 1
                aux_df[:,0] = azim_unique
                aux_df_noise[:,0] = azim_unique                   
                # random scan 
                t = np.random.randint(scanlist[0],scanlist[-1])
                ind_t = (scan.flatten() >=t) & (scan.flatten() < t+DT)
                x = r.flatten()*np.cos(azim.flatten()*np.pi/180)
                y = r.flatten()*np.sin(azim.flatten()*np.pi/180)
                X = np.c_[CNR.flatten()[ind_t],x[ind_t],y[ind_t]]
                transformer = RobustScaler(quantile_range=(25, 75))
                transformer.fit(X)
                X = transformer.transform(X)  
                kde = KernelDensity(kernel='gaussian', bandwidth=.021)
                kde.fit(X)
                samples = kde.sample(n_samples=45*198)
                samples = transformer.inverse_transform(samples)
                x_int = r_0_g*np.cos(phi_0_g)
                y_int = r_0_g*np.sin(phi_0_g)
                X_int = sp.interpolate.griddata(samples[:,1:],samples[:,0], (x_int.flatten(), y_int.flatten()), method='linear')
                samples_CNR = np.reshape(X_int,r_0_g.shape)                  
                for i in range(len(r_unique)):
                    #aux_df[:,n:n+3] = np.c_[r_0_g[:,i],vlos0[:,i],samples_CNR]
                    aux_df_noise[:,n:n+3] = np.c_[r_0_g[:,i],vlos0_noise[:,i],samples_CNR[:,i]]
                    n = n+3
                df_vlos0_noise = pd.concat([df_vlos0_noise, pd.DataFrame(data=aux_df_noise,columns=labels)])
                #df_vlos0 = pd.concat([df_vlos0, pd.DataFrame(data=aux_df,columns=labels)])
                count+=1
                print(count)
                
                
df_vlos0_noise['scan'] = df_vlos0_noise.groupby('azim').cumcount()


# Smoothing of CNR
df_vlos0_noise_med = df_vlos0_noise.copy()
#median filter in radial direction
df_vlos0_noise_med.CNR = df_vlos0_noise.CNR.rolling(15,axis=1, min_periods = 1).median()
#median filter in azimuth direction, not necessary
#df_vlos0_noise_med.CNR = df_vlos0_noise_med.CNR.rolling(3,axis=0, min_periods = 1).median()

# Contaminate CNR with noise-like values also!!

#with open('df_vlos_noise.pkl', 'wb') as writer:
#    pickle.dump(df_vlos0_noise,writer)

with open('df_vlos0_noise.pkl', 'rb') as reader:
    df_vlos0_noise = pickle.load(reader)

ind_scan = df_vlos0_noise.scan == 50

plt.figure()

plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_vlos0_noise.ws.loc[ind_scan ].values, 30, cmap='jet')

plt.figure()

plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df_vlos0_noise.CNR.loc[ind_scan ].values, 30, cmap='jet')


dfcopy = df.copy()

dfcopy = fl.df_ws_grad(dfcopy,grad='dvdr')

mask_CNR = ((df.CNR>-24) & (df.CNR<-8))

mask_CNR.columns = mask.columns

mask_tot = mask.copy()

mask_tot.mask(mask_CNR,other=False,inplace=True)

dfcopy.ws=df.ws.mask(mask_tot)

ind_scan = (df.scan >= 10) & (df.scan <= 90)

plt.figure()
im = plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df.ws.loc[ind_scan].values, 30, cmap='jet')
plt.colorbar(im)

#plt.figure()
#im = plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), dfcopy.ws.loc[ind_scan].values, 30, cmap='jet')
#plt.colorbar(im)

plt.figure()
im = plt.contourf(r_0_g*np.cos(phi_0_g),r_0_g*np.sin(phi_0_g), df.CNR.loc[ind_scan].values, 30, cmap='jet')
plt.colorbar(im)

########### Noise CNR
#########################

noisy_CNR = df.CNR.loc[ind_scan].values.flatten()

noisy_CNR = noisy_CNR[(noisy_CNR<-24)&(noisy_CNR>-40)]

transformer = RobustScaler(quantile_range=(25, 75))
transformer.fit(noisy_CNR.reshape(-1, 1))
X = transformer.transform(noisy_CNR.reshape(-1, 1))

from sklearn.model_selection import GridSearchCV
bandwidths = 10**np.linspace(-2, 0, 20)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=2)
grid.fit(X)

kde_CNR = sp.stats.gaussian_kde(noisy_CNR, bw_method='silverman')
CNR_noise = kde_CNR.resample(size=None)[source]

plt.figure()
plt.hist(noisy_CNR,bins=100)

###########################


ind_scan = (df.scan > 9000) & (df.scan < 12000)
ind_filtered = np.isnan(dfcopy.ws.loc[ind_scan].values)
ind_non_filtered = ~np.isnan(dfcopy.ws.loc[ind_scan].values)

filtered_CNR = df.CNR.loc[ind_scan].values[ind_filtered]
non_filtered_CNR = df.CNR.loc[ind_scan].values[ind_non_filtered]

ind_CNR = filtered_CNR<-24#~((filtered_CNR>-5) & (filtered_CNR<-30))

filtered_ws = df.ws.loc[ind_scan].values[ind_filtered]
non_filtered_ws = df.ws.loc[ind_scan].values[ind_non_filtered]

filtered_dvdr = dfcopy.loc[ind_scan].dvdr.values[ind_filtered]
non_filtered_dvdr = dfcopy.loc[ind_scan].dvdr.values[ind_non_filtered]

plt.figure()
plt.hist(filtered_ws[ind_CNR],bins=30)
plt.figure()
plt.hist(filtered_CNR[ind_CNR],bins=30)
plt.figure()
plt.hist(np.abs(filtered_dvdr[ind_CNR]),bins=30)

plt.figure()
plt.hist(non_filtered_ws[ind_CNR],bins=30)
plt.figure()
plt.hist(non_filtered_CNR[ind_CNR],bins=30)
plt.figure()
plt.hist(np.abs(non_filtered_dvdr),bins=30)

plt.figure()
plt.scatter(np.abs(filtered_dvdr[ind_CNR]),filtered_CNR[ind_CNR])

# In[]
#############################################

#####################Numerical lidar, figure

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1300, 1000)
ax.set_ylim(-5500,-3500)
ax.set_zlim(0, 7)

plt.show()
r = 5000
r0 = r
phi0 = -30
dl = 75
for i in range(3):
    phi_0 = phi0   
    for j in range(4):        
        wedge = Wedge((-4000,-3000), r, phi_0, phi_0+5, width=600, fill = False, edgecolor = 'k', linewidth = 1,alpha=.3)
        ax.add_patch(wedge)
        art3d.pathpatch_2d_to_3d(wedge, z=0, zdir="z")
        phi_0+=5
    r = r-600

wedge = Wedge((-4000,-3000), r0, phi0, phi_0, width= r0-r, facecolor ='lightgrey', edgecolor = 'k', linewidth = 1, zorder=10)    

x_int = np.arange(-2000, 2000, 100)
y_int = np.arange(-6000, -3000, 100)
g = np.meshgrid(x_int, y_int)
coords = list(zip(*(c.flat for c in g)))

lim_x = np.array([np.cos(-20*np.pi/180)*3800,np.cos(-20*np.pi/180)*4400])-4000
lim_y = np.array([np.sin(-20*np.pi/180)*3800,np.sin(-20*np.pi/180)*4400])-3000

line_x = np.linspace(lim_x[0],lim_x[1],51)
line_y = np.linspace(lim_y[0],lim_y[1],51) 

h = 2*(line_x-line_x[int(len(line_x)/2)])/(lim_x[1]-lim_x[0])

delta_r = 35
#aux_1 = np.reshape(np.repeat(r_unique,len(r_refine),axis = 0),(len(r_unique),len(r_refine)))
#aux_2 = np.reshape(np.repeat(r_refine,len(r_unique)),(len(r_refine),len(r_unique))).T
r_F = h*(lim_x[1]-lim_x[0])/4#aux_1-aux_2
rp = dl/(2*np.sqrt(np.log(2)))

erf = sp.special.erf((r_F+.5*delta_r)/rp)-sp.special.erf((r_F-.5*delta_r)/rp)
w = (1/2/delta_r)*erf
#w = .75*(1-h**2)
w = 500*w

points = np.vstack([p for p in coords if wedge.contains_point(p, radius=0)])

ax.scatter(points[:,0],points[:,1],np.zeros(points[:,0].shape),'.',c='k',s=3)

ax.grid(False)
ax.set_frame_on(False)

ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

# Get rid of the spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# Get rid of the panes
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

for an in [-10,-15,-25,-30]:
    a_x_s = np.array([np.cos(an*np.pi/180)*2600,np.cos(an*np.pi/180)*5600])-4000
    a_y_s = np.array([np.sin(an*np.pi/180)*2600,np.sin(an*np.pi/180)*5600])-3000
    a_s = Arrow3D(a_x_s, a_y_s, [0, 0], mutation_scale=20, lw=2, arrowstyle="-|>", linestyle = '--', color="grey",alpha=.5)
    ax.add_artist(a_s)

ax.plot(line_x,line_y,w,c='r',lw=2)

h = 0 
v = []
for k in range(0, len(line_x) - 1):
    x = [line_x[k], line_x[k+1], line_x[k+1], line_x[k]]
    y = [line_y[k], line_y[k+1], line_y[k+1], line_y[k]]
    z = [w[k], w[k+1], h, h]
    v.append(list(zip(x, y, z)))
poly3dCollection = Poly3DCollection(v,facecolor='red',alpha=.1)
ax.add_collection3d(poly3dCollection)

a_x = np.array([np.cos(-20*np.pi/180)*2600,np.cos(-20*np.pi/180)*5600])-4000
a_y = np.array([np.sin(-20*np.pi/180)*2600,np.sin(-20*np.pi/180)*5600])-3000
a = Arrow3D(a_x, a_y, [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color="b")
ax.add_artist(a)

rangesx = np.array([np.cos(-20*np.pi/180)*(2900 + i*300) for i in range(9)])-4000
rangesy = np.array([np.sin(-20*np.pi/180)*(2900 + i*300) for i in range(9)])-3000
ax.scatter(rangesx,rangesy,np.zeros(len(rangesx)),'.',c='b',s=30)
fig.tight_layout()
ax.text2D(0.63, 0.8, '$w$', transform=ax.transAxes,fontsize=30)#\:=\:0.75(1-h^2)$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.72, 0.4, '$Laser\:beam$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.2, 0.2, '$Range\:gate$', transform=ax.transAxes,fontsize=30)
ax.text2D(0.22, 0.75, '$Beams\:to\:be\:averaged$', transform=ax.transAxes,fontsize=30)
#ax.text2D(0.15, 0.85, '(a)', transform=ax.transAxes,fontsize=24)
a_w = Arrow3D([rangesx[6],rangesx[4]], [rangesy[6],rangesy[4]], [8, w[25]],
              mutation_scale=20, lw=3, arrowstyle="wedge", color="k")
ax.add_artist(a_w)

a_b = Arrow3D([rangesx[8]+150,rangesx[7]], [rangesy[8]+150,rangesy[7]], [3, 0],
              mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
ax.add_artist(a_b)

a_r = Arrow3D([-500,rangesx[4]], [-5200,rangesy[4]], [0, 0],
              mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
ax.add_artist(a_r)

for an in [-10,-15,-25,-30]:
    a_x_s = np.array([np.cos(an*np.pi/180)*2600,np.cos(an*np.pi/180)*5600])-4000
    a_y_s = np.array([np.sin(an*np.pi/180)*2600,np.sin(an*np.pi/180)*5600])-3000

    a_f = Arrow3D([a_x[0],a_x_s[0]], [a_y[0],a_y_s[0]],[4, 0],
                  mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
    ax.add_artist(a_f)
################    





















