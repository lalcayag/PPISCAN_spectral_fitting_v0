# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:24:19 2018

@author: lalc
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from subprocess import Popen, PIPE

# In[]

#Sepctral tensor parameters

ae = .05
L = 1000
G = 3.1
N_x = 512
N_y = 512
L_x = 7000
L_y = 7000

x = np.linspace(0,L_x,N_x)
y = np.linspace(0,L_y,N_y)

input_file = 'sim.inp.txt'
file = open(input_file,'w') 
file.write('2\n') 
file.write('2\n') 
file.write('1\n')
file.write('2\n') 
file.write('3\n')  
file.write(str(N_x)+'\n')
file.write(str(N_y)+'\n')
file.write(str(L_x)+'\n')
file.write(str(L_y)+'\n')
file.write('basic\n')
file.write(str(ae)+'\n')
file.write(str(L)+'\n')
file.write(str(G)+'\n')
file.write('-1\n')
file.write('simu\n')
file.write('simv\n')
file.close() 

arg = 'windsimu'+' '+input_file
p=subprocess.run(arg)

u = np.reshape(np.fromfile("simu", dtype=np.float32),(N_x,N_y))

plt.figure()
plt.contourf(x,y,u,300,cmap='jet')
plt.colorbar()

v = np.reshape(np.fromfile("simv", dtype=np.float32),(N_x,N_y))

plt.figure()
plt.contourf(x,y,v,300,cmap='jet')
plt.colorbar()
