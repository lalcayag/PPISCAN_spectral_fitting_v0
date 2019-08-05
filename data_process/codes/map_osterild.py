# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:02:04 2019

@author: lalc
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pyproj import Proj, transform
from mpl_toolkits.basemap import pyproj
import numpy as np
from matplotlib.patches import Wedge, Circle
from matplotlib.ticker import EngFormatter, StrMethodFormatter

fig, ax = plt.subplots(1,2,figsize=(8, 6))
#ax.set_aspect('equal')
ax[0].use_sticky_edges = False
ax[0].margins(0.07)
ax[0].text(0.05, 0.95, '(a)', color='white', transform=ax[0].transAxes, fontsize=24,
        verticalalignment='top')

ax[1].use_sticky_edges = False
ax[1].margins(0.07)
ax[1].text(0.05, 0.95, '(b)', color='white', transform=ax[1].transAxes, fontsize=24,
        verticalalignment='top')

x1, y1 = 492768.8, 6322832.3
x2, y2 = 492768.7, 6327082.4
scale = 35
dx = scale*(x1-x2)
dy = scale*(y1-y2)
dx = .5*dy

myproj = Proj("+proj=utm +zone=32N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lon, lat = myproj([x1+dx,x2-dx], [y1+dy,y2-dy], inverse=True)

ax[0].set_xticks(np.linspace(lon[0]+.1, lon[1]-.1, 4))
ax[0].set_yticks(np.linspace(lat[0]+.1, lat[1]-.1, 4))

x1d, y1d = myproj(x1, y1, inverse=True)
x1dm, y1dm = myproj(.5*(x1+x2), .5*(y1+y2), inverse=True)
x2d, y2d = myproj(x2, y2, inverse=True)

mymap = Basemap(llcrnrlon = lon[0],llcrnrlat=lat[0],urcrnrlon = lon[1],
                urcrnrlat=lat[1], resolution='f', suppress_ticks=False, ax=ax)
mymap.ax = ax[0]
ptmid = mymap.plot(x1dm, y1dm, 'o',markersize=40, markerfacecolor= 'none', markeredgecolor= 'red', latlon=True)
mymap.arcgisimage(service='World_Imagery', xpixels = 3000, verbose= True)

scale = 2
dx = scale*(x1-x2)
dy = scale*(y1-y2)
dx = dy

myproj = Proj("+proj=utm +zone=32N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lon, lat = myproj([x1+dx,x2-dx], [y1+dy,y2-dy], inverse=True)

x1d, y1d = myproj(x1, y1, inverse=True)
x1dm, y1dm = myproj(.5*(x1+x2), .5*(y1+y2), inverse=True)
x2d, y2d = myproj(x2, y2, inverse=True)

mymap = Basemap(llcrnrlon = lon[0],llcrnrlat=lat[0],urcrnrlon = lon[1],
                urcrnrlat=lat[1], resolution='f', suppress_ticks=False, ax=ax)
mymap.ax = ax[1]
pt0 = mymap.plot([x1d, x2d], [y1d, y2d], 'ro', markersize=15, latlon=True, label = 'Meteo. masts and WindSanner location')
mymap.arcgisimage(service='World_Imagery', xpixels = 3000, verbose= True)

fig.set_size_inches(20,8)

ax[1].set_xticks(np.linspace(lon[0]+.01, lon[1]-.01, 4))
ax[1].set_yticks(np.linspace(lat[0]+.01, lat[1]-.01, 4))

ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].tick_params(axis='both', which='major', labelsize=16)


ax[0].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.2f} $^\circ$ N"))
ax[0].xaxis.set_major_formatter(StrMethodFormatter(u"{x:.2f} $^\circ$ E"))
ax[1].yaxis.set_major_formatter(StrMethodFormatter(u"{x:.2f} $^\circ$ N"))
ax[1].xaxis.set_major_formatter(StrMethodFormatter(u"{x:.2f} $^\circ$ E"))


myproj = Proj("+proj=utm +zone=32N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lonw, latw = myproj([x1,x1+7000], [y1,y1], inverse=True)

ax[1].legend(fontsize=22)

wedge = Wedge((x1d, y1d), np.diff(lonw), 106, 196, width=np.diff(lonw), color ='r',alpha=.25)
ax[1].add_artist(wedge)
wedge = Wedge((x2d, y2d), np.diff(lonw), 166, 256, width=np.diff(lonw), color ='b', alpha = 0.25)
ax[1].add_artist(wedge)
wedge = Wedge((x2d, y2d), np.diff(lonw), 106+180, 196+180, width=np.diff(lonw), color ='b',alpha=.25)
ax[1].add_artist(wedge)
wedge = Wedge((x1d, y1d), np.diff(lonw), 166+180, 256+180, width=np.diff(lonw), color ='r', alpha = 0.25)
ax[1].add_artist(wedge)
fig.tight_layout()
plt.savefig(r'C:\Users\lalc\Documents\Old Documents folder\Publications\Filtering using a clustering algorithm\Figures\balcony')

# lons, lats, xs, ys = mymap.makegrid(200, 200, returnxy=True)
# gc = pyproj.Geod(a=mymap.rmajor, b=mymap.rminor)
# distances1 = np.zeros(lons.size)
# distances2 = np.zeros(lons.size)
# for k, (lo, la) in enumerate(zip(lons.flatten(), lats.flatten())):
#     _, _, distances1[k] = gc.inv(x1d, y1d, lo, la)
#     _, _, distances2[k] = gc.inv(x2d, y2d, lo, la)
#    
# distances1 = distances1.reshape(200, 200)  # In km.
# distances2 = distances2.reshape(200, 200)  # In km.

# Plot perimeters of equal distance.
# levels = [1000]  # [50, 100, 150]
# cs1 = mymap.contour(xs, ys, distances1, levels, colors='k')
# cs2 = mymap.contour(xs, ys, distances2, levels, colors='k')
#
#mymap.arcgisimage(service='World_Shaded_Relief', xpixels = 3000, verbose= True)
#mymap.arcgisimage(service='World_Imagery', xpixels = 3000, verbose= True)
#mymap.arcgisimage(service='World_Topo_Map', xpixels = 3000, verbose= True)
#mymap.arcgisimage(service='Ocean_Basemap', xpixels = 3000, verbose= True)





#http://server.arcgisonline.com/arcgis/rest/services


myproj = Proj("+proj=utm +zone=32N, +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
scale = 2
dx = scale*(x1-x2)
dy = scale*(y1-y2)
dx = dy
lon, lat = myproj([x1+dx,x2-dx], [y1+dy,y2-dy], inverse=True)
x1d, y1d = myproj(x1, y1, inverse=True)
x2d, y2d = myproj(x2, y2, inverse=True)

lat_start, lat_end = lat[0], lat[1]
lon_start, lon_end = lon[0], lon[1]

names=['Østerild']

makeMap(lon_start, lon_end, lat_start, lat_end, names, [x1d,x2d], [y1d,y2d])


import sys
sys.path.append("/Users/lalc/dev/ext-libs")

import elevation

from elevation import cli
cli.selfcheck()

import elevation
import os
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, Resampling

bounds = np.array([lon[0], lat[0], lon[1], lat[1]])

bounds = [-90.000672,	45.998852,	-87.998534,	46.999431]

west, south, east, north = bounds
west, south, east, north = bounds  = west - .05, south - .05, east + .05, north + .05
dem_path = '\\Iron_River_DEM.tif'
output = os.getcwd() + dem_path
elevation.clip(bounds=bounds, output=output, product='SRTM3')





