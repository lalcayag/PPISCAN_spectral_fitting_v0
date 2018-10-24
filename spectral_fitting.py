<<<<<<< HEAD
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

import


# In[Data loading]
# To do: it is necessary to link the mask file with the source file
root = tkint.Tk()
file_spec = tkint.filedialog.askopenfilenames(parent=root,title='Choose a Spectra file')
root.destroy()
root = tkint.Tk()
file_dir = tkint.filedialog.askopenfilenames(parent=root,title='Choose an Output file')
root.destroy()

# In[Data loading]
with open(file_spec[0], 'rb') as S:
     pickle.dump((Su_u,Sv_v,Su_v,k_1,k_2),V_t)  
