# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:41:55 2018

Package for wind field reconstruction from PPI scans (it might be used also with other type of scan)

@author: 
Leonardo Alcayaga
lalc@dtu.dk

"""
# In[Packages used]

import numpy as np

from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.tri import Triangulation

# In[############# [Functions] #################]

# In[un-structured grid generation]

def translationpolargrid(mgrid,h):  
    """
    Function that performs a linear translation from (r,theta) = (x,y) -> (x0,y0) = (x+h[0],y+h[1])

    Input:
    -----
        mgrid                 - Tuple containing (rho,phi), polar coordinates to transform
        
        h                     - Linear distace of translation   
        
    Output:
    ------
        rho_prime, phi_prime  - Translated polar coordinates      
    """
    # Original polar coordinates
    rho = mgrid[0]
    phi = mgrid[1]
    # Trnasformation to cartesian
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    # Translation in cartesian
    x0 = x - h[0]
    y0 = y - h[1]
    # Transformation back to polar coordinates
    rho_prime = np.sqrt(x0**2+y0**2)
    phi_prime = np.arctan2(y0,x0)
    return(rho_prime, phi_prime)

# In[Nearest point]

def nearestpoint(mg0,mg1,dr,dp):
    """
    Function to identify the points inside the overlapping area of two PPI scans. A nearest neighbour
    approach is used, in polar coordinates.

    Input:
    -----
        mg0      - Tuple with (r0,p0), points in polar coordinates of the first scan in a common
                   frame. This means that the local PPI scan coordinates must be translated to a
                   common point with the other scans.
        
        mg1      - Tuple with (r1,p1), points in polar coordinates of the first scan in a common
                   frame. This means that the local PPI scan coordinates must be translated to a
                   common point with the other scans.
        
        dr       - Grid spacing in the r component in polar coordinates.
        
        dp       - Grid spacing in the azimuth component in polar coordinates.
        
    Output:
    ------
        nearest  - Tuple with (r,p,ind), namely, r, all the r coordinates of points within the
                   overlapping area, p, same for the azimuth coordinate and ind, the corresponding
                   index in the original flatten array of coordinates of one scan respect to 
                   the other.     
    """
    # Coordinates extraction. A common frame is assumed.
    r0 = mg0[0].flatten()
    r1 = mg1[0].flatten()
    p0 = mg0[1].flatten()
    p1 = mg1[1].flatten()
    # Initialization of list with nearest points to one scan.
    raux = []
    paux = []
    iaux = []
    for i in range(len(r1)):
        # Distances of all points in the two scans. In polar coordinates
        dist1 = np.sqrt((r0-r1[i])**2)
        dist2 = np.sqrt((p0-p1[i])**2)
        # Index of points in scan 1 within a neighbourhood dr x dp
        ind = ((dist1<=dr) & (dist2<=dp)).nonzero()[0] 
        # Append of those points
        raux.append(r0[ind])
        paux.append(p0[ind])
        iaux.append(ind)
    # Flatten list    
    r_flat= [item for sublist in raux for item in sublist]
    p_flat= [item for sublist in paux for item in sublist]
    i_flat= [item for sublist in iaux for item in sublist]
    # List with corresponding r and azimuth coordinates of nearest points, and corresponding index
    # in original array
    polar = zip(r_flat, p_flat, i_flat)
    # Unique values
    unique = [list(t) for t in zip(*list(set(polar)))] 
    # Final output
    nearest = (np.array(unique[0]),np.array(unique[1]),np.array(unique[2]))
    return nearest

# In[Overlapping grid]

def grid_over2(mg0, mg1, d):
    """
    Function to define coordinates (in a common frame) of the intersection of the laser-beams from
    two PPI scans. This function uses a kd-tree appoach that make the grid generation independent of 
    scan geometry, since finds intersection points only finding nearest neighbours between scans

    Input:
    -----
        mg0      - Tuple with (r0,p0), points in polar coordinates of the first scan in local frame
                   and non translated.
        
        mg1      - Tuple with (r1,p1), points in polar coordinates of the first scan in local frame
                   and non translated.
        
        d        - Linear distance between LiDARs.
        
    Output:
    ------
        tree_g   - Kd-tree of laser beam intersection points
        
        tri      - Unstructured grid from Delaunay triangulation with triangles corners as
                   intersection points
        
        w0       - Weights to be used in wind field reconstruction, depending on distance of 
                   current scan 0 point to the nearest intersection point.
        
        n0       - Corresponding label of scan-0 points within the neighbourhood of each 
                   intersection point.
        
        i_o_0    - Original index of neigbouring points in scan 0.
        
        w1       - Same weights but for scan 1
        
        n1       - Same labels bit for scan1 
        
        i_o_1    - Same index but this time for scan 1
    
    """
    # Polar grid resolution is used as nearest neighbour distance to estimate
    # overlaping scanning area
    dr = min(np.diff(np.unique(mg0[0].flatten())))/2
    dp = min(np.diff(np.unique(mg0[1].flatten())))/2   
    # Translation of grids
    r0, p0 = translationpolargrid(mg0,-d/2)
    r1, p1 = translationpolargrid(mg1,d/2) 
    # Overlapping points
    r_o_0, p_o_0, i_o_0 = nearestpoint((r0,p0),(r1,p1),dr,dp)
    r_o_1, p_o_1, i_o_1 = nearestpoint((r1,p1),(r0,p0),dr,dp)
    # Cartesian trees from overlapping points of each scan
    pos0 = np.c_[r_o_0*np.cos(p_o_0),r_o_0*np.sin(p_o_0)]    
    tree_0 = KDTree(pos0)
    pos1 = np.c_[r_o_1*np.cos(p_o_1),r_o_1*np.sin(p_o_1)]    
    tree_1 = KDTree(pos1)  
    # Intersection points, first iteration will find pair of points
    ind0,dist0 = tree_0.query_radius(tree_1.data, r=3*dr/2,return_distance = True,sort_results=True)
    ind1,dist1 = tree_1.query_radius(tree_0.data, r=3*dr/2,return_distance = True,sort_results=True) 
    ind00 = []
    ind01 = []
    for i,j in zip(ind0,range(len(ind0))):
        if i.size > 0:
            #indices
            ind00.append(np.asscalar(i[0]))
            ind01.append(j)           
    ind10 = []
    ind11 = []
    for i,j in zip(ind1,range(len(ind1))):
        if i.size > 0:
            #indices
            ind11.append(np.asscalar(i[0]))
            ind10.append(j)      
    # Center of grafity of near-intersection points
    posg0=np.c_[0.5*(pos0[:,0][ind00]+pos1[:,0][ind01]),0.5*(pos0[:,1][ind00]+pos1[:,1][ind01])] 
    posg1=np.c_[0.5*(pos0[:,0][ind10]+pos1[:,0][ind11]),0.5*(pos0[:,1][ind10]+pos1[:,1][ind11])]  
    posg = np.vstack((posg0,posg1))
    unique = [list(t) for t in zip(*list(set(zip(posg[:,0], posg[:,1]))))] 
    posg = np.c_[unique[0],unique[1]]
    tree_g = KDTree(posg)  
    # Intersection points, final iteration 
    # Identification of nearest neighbours to each preestimated intersection point
    indg, distg = tree_g.query_radius(tree_g.data, r=2*dr, return_distance = True, sort_results=True)
    S = sorted(set((tuple(sorted(tuple(i))) for i in indg if len(tuple(i))>1)))
    nonS = [np.asscalar(i) for i in indg if len(tuple(i))==1]
    temp = [set(u) for u in S]
    S = []
    for ti in temp:
        aux = [t for t in temp if t!=ti]
        if not any(ti <= u for u in aux):
            S.append(list(ti))
    aux=np.array([np.mean(posg[list(p),:],axis=0) for p in S])  
    posg = np.vstack((aux,posg[nonS])) 
    tree_g = KDTree(posg)
    # Diastances and labels of neighbours to intersection points
    d0,n0 = tree_g.query(tree_0.data, return_distance = True)
    d1,n1 = tree_g.query(tree_1.data, return_distance = True)
    # Weights estimation
    w0 = np.squeeze(d0)**-1
    w1 = np.squeeze(d1)**-1
    # Correct dimensions!
    n0 = np.squeeze(n0)
    n1 = np.squeeze(n1)
    # Delaunay triangulation of intersection points
    tri = Triangulation(posg[:,0], posg[:,1])  
    return (tree_g, tri, w0, n0, i_o_0, w1, n1, i_o_1)

# In[Wind field reconstruction]

def wind_field_rec(Lidar0, Lidar1, tree, triangle, d):
    """
    Function to reconstruct horizontal wind field (2D) in Cartesian coordinates, taking advantage of
    Kd-tree structures. This function works with PPI scans that are not synchronous and using 
    equation (12) in [1]. Continuity might be included in this formulation, and uncertainty 
    estimation.

    Input:
    -----
        Lidar_i  - Tuple with (vr_i,r_i,phi_i,w_i,neigh_i,index_i):
            
                        vr_i          - Array with V_LOS of Lidar_i
                        r_i, phi_i    - Arrays with polar coordinates of the first scan in local 
                                        frame and non translated.
                        w_i           - Array with weights of each measurement vr_i dependant on
                                        distance from (r_i, phi_i) to the correponding unstructured 
                                        grid point in triangle.
                        neigh_i       - Array with indexes of the corresponding nearest intersection
                                        point.
                        index_i       - Array with indexes of the original polar grid in local Lidar
                                        coordinates.
        
        tree     - Kd-tree of the unstructured grid of laser beams intersection points.
        
        triangle - Delaunay triangulation with the unstructured grid of laser beams intersection
                   points.
        
        d        - Linear distance between Lidar_i and Lidar_j.
        
    Output:
    ------
        U, V     - Cartesian components of wind speed.
    
    
    [1] Michel Chong and Claudia Campos, Extended Overdetermined Dual-Doppler Formalismin 
        Synthesizing Airborne Doppler Radar Data, 1996, Journal of Atmopspheric and Oceanic Technology
    """
    # Input extraction
    vr0, phi0_old, w0, neigh0, index0 = Lidar0
    vr1, phi1_old, w1, neigh1, index1 = Lidar1 
    vr0 = vr0.values.flatten()[index0]
    vr1 = vr1.values.flatten()[index1]
    phi0_old = phi0_old.flatten()[index0]
    phi1_old = phi1_old.flatten()[index1]
    # Initialization of wind components
    U = np.ones(len(tree.data))
    V = np.ones(len(tree.data))
    U[U==1] = np.nan
    V[V==1] = np.nan
    # Loop over each member of the un-structured grid tree
    for i in range(len(tree.data)):
        # Identification of neighbouring observations to intersection points.
        ind0 = (neigh0==i).nonzero()[0]
        ind1 = (neigh1==i).nonzero()[0]
        # Selection of valid observations only
        ind00 = (~np.isnan(vr0[ind0])).nonzero()[0]
        ind11 = (~np.isnan(vr1[ind1])).nonzero()[0]
        # Corresponding V_LOS
        vr_0 = vr0[ind0][ind00]
        vr_1 = vr1[ind1][ind11]
        # Corresponding weights
        w_0  = w0[ind0][ind00]
        w_1  = w1[ind1][ind11]
        # Transformation to cartesian coordinates
        sin0 = np.sin(phi0_old[ind0][ind00])
        cos0 = np.cos(phi0_old[ind0][ind00]) 
        sin1 = np.sin(phi1_old[ind1][ind11])
        cos1 = np.cos(phi1_old[ind1][ind11])
        # LSQ fitting of all observations in the neighbourhood, only if it has valid observations 
        # from at least two Lidar's
        if (w_0.size > 0) and (w_1.size > 0):
            # Coefficients of linear equation system, including weights
            alpha_i = np.r_[sin0*w_0,sin1*w_1]
            beta_i = np.r_[cos0*w_0,cos1*w_1]
            V_i = np.r_[vr_0*w_0,vr_1*w_1]
            # Components in matrix of coefficients
            S11 = np.nansum(alpha_i**2)
            S12 = np.nansum(alpha_i*beta_i)
            S22 = np.nansum(beta_i**2)
            # V_LOS solution vector
            V11 = np.nansum(alpha_i*V_i)
            V22 = np.nansum(beta_i*V_i)
            # Coeffiecient matrix
            a = np.array([[S11,S12], [S12,S22]])
            b = np.array([V11,V22])
            # Linear system solution
            x = np.linalg.solve(a, b)
            U[i] = x[0]
            V[i] = x[1]
        else:
            # Grid points lacking information from one or both Lidars are assumed as NaN.
            U[i], V[i] = np.nan, np.nan
    return (U, V)

# In[]   
def data_interp_triang(U,V,x,y,dt): 
    """
    Function to interpolate wind speed in grid points with missing information from both or one Lidar.
    Kd-tree is again used to do the regression and interpolation, this time neighbours are defined 
    in space and time, the latter assuming a constant wind speed and trajectory in successive scans.

    Input:
    -----
        U, V          - Lists of arrays representing the wind field in cartesian coordinates in each 
                        grid point of a triangulation represented by coordinates x and y. Each array
                        in the list represent  one scan.
                       
        x, y          - Arrays with cartesian coordinates of the un-structured grid.
        
        dt            - Time step between scans.
        
    Output:
    ------
        U_int,V_int   - List of arrays with interpolated wind speed field. Each array in the list
                        represent  one scan.
       
    """  
    # Initialization of the kdtree regressor. The number of neighbours, n_neighbours is set equal to
    # the number of corners and midpoints of a cube
    neigh = KNeighborsRegressor(n_neighbors=26,weights='distance',algorithm='auto', leaf_size=30,
                                n_jobs=1) 
    # Initialization of output
    U_int = []
    V_int = []
    it = range(len(U))
    # Loop over the list elements
    for scan in it:
        print(scan)
        # Initial and final scans are interpolated only in one direction in time, orwards and 
        # backwards, respectively.
        if scan == it[0]:
            # Temporal coordinate to spatial, assuming constant speed between scans.
            xj = x-U[scan+1]*dt
            yj = y-V[scan+1]*dt
            # Input for kd-regressor
            X = np.c_[np.r_[x,xj],np.r_[y,yj]]
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()
            # Indexes of missing wind speeds in current and next scan
            ind0 = np.isnan(np.r_[U[scan],U[scan+1]])
            # Indexes in current scan
            ind1 = np.isnan(U[scan])
            # Check if there are any missing wind speed in the current scan and if there are enough
            # neighbours for interpolation.
            if (sum(ind1)>0) & (sum(~ind0)>26):
                # Regressor is defined for the actual scan
                neigh.fit(X[~ind0,:], np.r_[U[scan],U[scan+1]][~ind0])
                # Interpolation is carried out
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                # Same for V component
                neigh.fit(X[~ind0,:], np.r_[V[scan],V[scan+1]][~ind0])
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                U_int.append(U_aux)
                V_int.append(V_aux)
            # If there are not enough neighbours.    
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                
            # If there is not missing data, go to the next scan   . 
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux)                
        # Same as before, this time when the last scan is reached.
        if scan == it[-1]:
            # Temporal coordinate to spatial, assuming constant speed between scans, this time going
            # backwards.
            xj = x+U[scan-1]*dt
            yj = y+V[scan-1]*dt 
            X = np.c_[np.r_[xj,x],np.r_[yj,y]]       
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()
            ind0 = np.isnan(np.r_[U[scan-1],U[scan]])
            ind1 = np.isnan(U[scan])        
            if (sum(ind1)>0) & (sum(~ind0)>26):               
                neigh.fit(X[~ind0,:], np.r_[U[scan-1],U[scan]][~ind0])              
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])                
                neigh.fit(X[~ind0,:], np.r_[V[scan-1],V[scan]][~ind0])               
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])              
                U_int.append(U_aux)
                V_int.append(V_aux)           
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                              
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux) 
        # All the rest, interpolated backwards and forwards in time.        
        else:
            # Temporal coordinate to spatial, forwards.
            xj = x+U[scan-1]*dt
            yj = y+V[scan-1]*dt
            # Temporal coordinate to spatial, backwards.
            xk = x-U[scan+1]*dt
            yk = y-V[scan+1]*dt
            X = np.c_[np.r_[xj,x,xk],np.r_[yj,y,yk]]
            U_aux = U[scan].copy()
            V_aux = V[scan].copy()
            ind0 = np.isnan(np.r_[U[scan-1],U[scan],U[scan+1]])
            ind1 = np.isnan(U[scan])
            if (sum(ind1)>0) & (sum(~ind0)>26):
                neigh.fit(X[~ind0,:], np.r_[U[scan-1],U[scan],U[scan+1]][~ind0])
                U_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                neigh.fit(X[~ind0,:], np.r_[V[scan-1],V[scan],V[scan+1]][~ind0])
                V_aux[ind1] = neigh.predict(np.c_[x,y][ind1,:])
                U_int.append(U_aux)
                V_int.append(V_aux)   
            if ~(sum(~ind0)>26):
                U_int.append([])
                V_int.append([])                  
            if ~(sum(ind1)>0):
                U_int.append(U_aux)
                V_int.append(V_aux) 
    return (U_int,V_int)