# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:27:46 2018

Module for spectral fitting based on:
    
    - Least-Squares with specific weights
    - Maximum Likelihood estimation
    - Markov-Chain Monte-Carlo

Autocorrelation is calculated in terms 

@author: lalc
"""
# In[Packages used]

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import os, sys
import re
import emcee
import corner
from scipy.stats import chi

#import matplotlib.pyplot as plt
#import numpy as np
#from matplotlib.tri import Triangulation,UniformTriRefiner,CubicTriInterpolator,LinearTriInterpolator,TriFinder,TriAnalyzer
                           
from sklearn.neighbors import KDTree
# In[Log-Likelihood]

def LLH(param,args=()):
    #args = (spectra_error,model,noise,k_1,F_obs)
    #print(param[-1],-1*sum(np.log(args[0](param,args=(args[1:])))))
    plt.plot(args[3]*(2*np.pi),args[1](param[:10],args=(args[3],))+args[2](param[11:13],args=(args[3],)),'-o')
    plt.plot(args[3]*(2*np.pi),args[-1],'--')
    plt.xscale('log')
    plt.yscale('log')
    return -1*sum(np.log(args[0](param,args=(args[1:]))))

# In[Turbulence Spectra models]

def spectra_peltier(param,args=()): 
    # Peltier 1996, Horizontal wind speed spectrum model
    #param = [c1,c2,l,s]
    #args = (k)
    # mixture of free conection and neutral conditions

    c1_f,c2_f,l_f,s_f, c1_n,c2_n,l_n,s_n,w,n= param 
    
    k_1 = args[0]
    E = lambda c1,c2,l,s,k: .71*c1*l**2*k*s**2/(c2+(k*l)**2)**(4/3)
    F = lambda c1,c2,l,s,k: .71*c1*l*s**2/((c2+(k*l)**2)**(5/6))
    E_m = E(c1_f,c2_f,l_f,s_f,k_1)+E(c1_n,c2_n,l_n,s_n,k_1)
    F_m = F(c1_f,c2_f,l_f,s_f,k_1)+F(c1_n,c2_n,l_n,s_n,k_1)
    H = (np.sin(k_1*(w*np.pi)/np.max(k_1))/(k_1*(w*np.pi)/np.max(k_1)))**n
    F_m = F_m*H
    return F_m

# In[Theoretical Spectra]
def spectra_noise(param,args=()):
    #args = (k)
#    return param[0]*args[0]**param[1]
    return np.exp(param[0]*args[0]**param[1])

# In[Noise in spectra]
def spectra_theo(param,args=()):
    #args = (F_model,F_noise,k)
    k_1 = args[2]
    param_noise = param[-2:]
    param_model = param[:-2]
    F_noise = args[1](param_noise,args= (k_1,))
    F_model = args[0](param_model,args= (k_1,))

    return (F_noise+F_model)

# In[Error distribution: Chi squared]

def spectra_error(param,args=()):
    #args = (F_model,F_noise,k,F_obs)
    args_theo = args[:-1]
    F_obs = args[-1]
    k_1 = args[-2]
    print(param)
    param_chi=param[-1]
    param_theo = param[:-1]
    F_theo = spectra_theo(param_theo,args=args_theo)
    #Chi-squared error
    error = param_chi*F_obs/F_theo
    return chi.pdf(error, param_chi) 
    
# In[Spectral fitting]
    
def spectra_fitting(F_obs,model,noise,k_1,param_init = [],bounds= []):

    if ~(len(param_init)>0):
        # Peltier
        param_init = [.85,23,2000,5, 1.6,1000,200,.36,2,4,0.001,2,2]
    
    if ~(len(bounds)>0):
        bound = [(0.5,10),(0,10),(500,2000),(0,10), (0,10),(0,2),(100,300),(0,1),(5/6,10),(0,.1),(0,3),(2,10)]
#    print('{0:9s}   {1:9s}   {2:9s}   {3:9s}   {3:9s}'.format('l_f', 's_f', 'l_n', 's_n', 'exp'))    
#    res = sp.optimize.minimize(LLH, param_init, args=((spectra_error,model,noise,k_1,F_obs),),method='L-BFGS-B', bounds = bound,callback=callbackF, options={'disp': True})
    print('{0:9s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('l_f', 's_f', 'l_n', 's_n', 'exp'))    
    res = sp.optimize.fmin_bfgs(LLH, param_init, args=((spectra_error,model,noise,k_1,F_obs),),callback=callbackF)
    return res
    
# In[]    
def callbackF(Xi):
    print('{0: 3.6f}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f} {4: 3.6f}'.format(Xi[2], Xi[3], Xi[6], Xi[7], Xi[8]))  
    
    
    
    
    