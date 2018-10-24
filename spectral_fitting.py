
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:54:15 2018

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

import spectralfitting as sf


# In[Data loading]
# To do: it is necessary to link the mask file with the source file
root = tkint.Tk()
file_spec_path = tkint.filedialog.askopenfilenames(parent=root,title='Choose a Spectra file')
root.destroy()
root = tkint.Tk()
file_out_path = tkint.filedialog.askopenfilenames(parent=root,title='Choose an Output file')
root.destroy()

# In[Data loading]
with open(file_spec_path[0], 'rb') as S:
     Su_u,Sv_v,Su_v,k_1,k_2 = pickle.load(S)

# In[Spectra fitting]
scan_n = 13000
F_obs = .5*(Su_u[scan_n,:] + Sv_v[scan_n,:])
res = spectra_fitting(F_obs[k_1[scan_n,:]>0],spectra_peltier,spectra_noise,
                      k_1[scan_n,k_1[scan_n,:]>0]) 