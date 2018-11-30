# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:27:46 2018

Module for 2D autocorrelation and spectra from horizontal wind field  
measurements. The structure expected is a triangulation from
scattered positions.

Autocorrelation is calculated in terms 

To do:
    
- Lanczos interpolaton on rectangular grid (usampling)

@author: lalc
"""
# In[Packages used]
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.tri import Triangulation,UniformTriRefiner,CubicTriInterpolator,LinearTriInterpolator,TriFinder,TriAnalyzer

from matplotlib import ticker    
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable  
                
from sklearn.neighbors import KDTree

# In[Autocorrelation for non-structured grid, brute force for non-interpolated wind field]  

def spatial_autocorr(tri,U,V,N,alpha):
    """
    Function to estimate autocorrelation in cartesian components of wind
    velocity, U and V. The 2D spatial autocorrelation is calculated in a 
    squared and structured grid, and represent the correlation of points
    displaced a distance tau and eta in x and y, respectively.

    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
        
        U,V      - Arrays with cartesian components of wind speed.
        
        N        - Number of points in the autocorrelation's squared grid.
        
        alpha    - Fraction of the spatial domain that will act as the limit 
                   for tau and eta increments. 
        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.  
                   
    """  
    # Squared grid of spatial increments
    tau = np.linspace(-alpha*(np.max(tri.x)-np.min(tri.x)),alpha*(np.max(tri.x)-np.min(tri.x)),N)
    eta = np.linspace(-alpha*(np.max(tri.y)-np.min(tri.y)),alpha*(np.max(tri.y)-np.min(tri.y)),N)
    tau,eta = np.meshgrid(tau,eta)
    # De-meaning of U and V. The mean U and V in the whole scan is used
    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    # Interpolator object to estimate U and V fields when translated by
    # (tau,eta)
    U_int = LinearTriInterpolator(tri, U)
    V_int = LinearTriInterpolator(tri, V)
    # Autocorrelation is calculated just for non-empty scans 
    if len(U[~np.isnan(U)])>0:
        # autocorr() function over the grid tau and eta.
        r_u = [autocorr(tri,U,U_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
        r_v = [autocorr(tri,V,V_int,t,e) for t, e in zip(tau.flatten(),eta.flatten())]
    else:
        r_u = np.empty(len(tau.flatten()))
        r_u[:] = np.nan
        r_v = np.empty(len(tau.flatten()))
        r_v[:] = np.nan
    
    return(r_u,r_v)

def autocorr(tri,U,Uint,tau,eta):
    """
    Function to estimate autocorrelation from a single increment in cartesian
    coordinates(tau, eta)
    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
        
        U        - Arrays with a cartesian component wind speed.
        
        U_int    - Linear interpolator object.
        
        tau      - Increment in x coordinate. 
        
        eta      - Increment in y coordinate.
        
    Output:
    ------
        r        - Autocorrelation function value.  
                   
    """ 
    # Only un-structured grid with valid wind speed
    ind = ~np.isnan(U)
    # Interpolation of U for a translation of the grid by (tau,eta)
    U_delta = Uint(tri.x[ind]+tau,tri.y[ind]+eta)
    # Autocorrelation on valid data in the original unstructured grid and the
    # displaced one.
    if len(U_delta.data[~U_delta.mask]) == 0:
        r = np.nan
    else:
        # Autocorrelation is the off-diagonal value of the correlation matrix.
        r = np.corrcoef(U_delta.data[~U_delta.mask],U[ind][~U_delta.mask],
                        rowvar=False)[0,1]
    return r


# In[Autocorrelation for non-structured grid, using FFT for interpolated
#                                                         (or not) wind field] 
    
def spatial_autocorr_fft(tri,U,V,N_grid=512,auto=False,transform = False, tree = None, interp = 'Lanczos'):
    """
    Function to estimate autocorrelation from a single increment in cartesian
    coordinates(tau, eta)
    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
                   
        N_grid     - Squared, structured grid resolution to apply FFT.
        
        U, V     - Arrays with cartesian components of wind speed.
        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.               
    """   
#    fig, ax = plt.subplots()
#    ax.set_aspect('equal')
#    ax.use_sticky_edges = False
#    ax.margins(0.07)
#    mask=TriAnalyzer(tri).get_flat_tri_mask(.05)
#    trid=Triangulation(tri.x,tri.y,triangles=tri.triangles[~mask])
#    grid = np.meshgrid(np.linspace(np.min(trid.x),np.max(trid.x),N_grid),np.linspace(np.min(trid.y),np.max(trid.y),N_grid)) 
#    V_intd= CubicTriInterpolator(trid, V)(grid[0].flatten(),grid[1].flatten()).data
#    V_intd = np.reshape(V_intd,grid[0].shape)
#    V_intd[np.isnan(V_intd)] = 0.0
#    ax.triplot(trid, color='black',lw=.5)
#    im = ax.contourf(grid[0],grid[1],V_intd,np.linspace(7.369614601135254,22.200645446777344,300),cmap='jet')
#    fig.colorbar(im)
    if transform:
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
        # Wind direction
        gamma = np.arctan2(V_mean,U_mean)
        # Components in matrix of coefficients
        S11 = np.cos(gamma)
        S12 = np.sin(gamma)
        T = np.array([[S11,S12], [-S12,S11]])
        vel = np.array(np.c_[U,V]).T
        vel = np.dot(T,vel)
        X = np.array(np.c_[tri.x,tri.y]).T
        X = np.dot(T,X)
        U = vel[0,:]
        V = vel[1,:]
        tri = Triangulation(X[0,:],X[1,:])
        mask=TriAnalyzer(tri).get_flat_tri_mask(.05)
        tri=Triangulation(tri.x,tri.y,triangles=tri.triangles[~mask])
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
    else:
        # Demeaning
        U_mean = avetriangles(np.c_[tri.x,tri.y], U, tri)
        V_mean = avetriangles(np.c_[tri.x,tri.y], V, tri)
        
    grid = np.meshgrid(np.linspace(np.min(tri.x),
           np.max(tri.x),N_grid),np.linspace(np.min(tri.y),
                 np.max(tri.y),N_grid))   
    
    U = U-U_mean
    V = V-V_mean
    
    
    #print(U_mean,avetriangles(np.c_[tri.x,tri.y], U, tri))
    # Interpolated values of wind field to a squared structured grid
    
#    U_int= CubicTriInterpolator(tri, U, kind='geom')(grid[0].flatten(),grid[1].flatten()).data
#    V_int= CubicTriInterpolator(tri, V, kind='geom')(grid[0].flatten(),grid[1].flatten()).data    
                  
    if interp == 'cubic':     
        U_int= CubicTriInterpolator(tri, U)(grid[0].flatten(),grid[1].flatten()).data
        V_int= CubicTriInterpolator(tri, V)(grid[0].flatten(),grid[1].flatten()).data 
        U_int = np.reshape(U_int,grid[0].shape)
        V_int = np.reshape(V_int,grid[0].shape)
    else:
#        U_int= lanczos_int_sq(grid,tree,U)
#        V_int= lanczos_int_sq(grid,tree,V) 
        U_int= LinearTriInterpolator(tri, U)(grid[0].flatten(),grid[1].flatten()).data
        V_int= LinearTriInterpolator(tri, V)(grid[0].flatten(),grid[1].flatten()).data 
        U_int = np.reshape(U_int,grid[0].shape)
        V_int = np.reshape(V_int,grid[0].shape)
    

               
    #zero padding
    U_int[np.isnan(U_int)] = 0.0
    V_int[np.isnan(V_int)] = 0.0
    #plt.triplot(tri)
    #plt.contourf(U_int,cmap='jet')
    fftU = np.fft.fft2(U_int)
    fftV = np.fft.fft2(V_int)
    if auto:

        # Autocorrelation
        r_u = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftU)**2)))/len(U_int.flatten())
        r_v = np.real(np.fft.fftshift(np.fft.ifft2(np.absolute(fftV)**2)))/len(U_int.flatten())
        r_uv = np.real(np.fft.fftshift(np.fft.ifft2(np.real(fftU*np.conj(fftV)))))/len(U_int.flatten())
    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    
    n = grid[0].shape[0]
    m = grid[1].shape[0]   
    # Spectra
    fftU  = np.fft.fftshift(fftU)
    fftV  = np.fft.fftshift(fftV) 
    fftUV = fftU*np.conj(fftV) 
    Suu = 2*(np.abs(fftU)**2)*(dx*dy)/(n*m)
    Svv = 2*(np.abs(fftV)**2)*(dx*dy)/(n*m)
    Suv = 2*np.real(fftUV)*(dx*dy)/(n*m)
#    kx = 1/(2*dx)
#    ky = 1/(2*dy)   
#    k1 = kx*np.linspace(-1,1,len(Suu))
#    k2 = ky*np.linspace(-1,1,len(Suu))
    
    k1 = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
    k2 = np.fft.fftshift((np.fft.fftfreq(m, d=dy)))
    if auto:
        return(r_u,r_v,r_uv,Suu,Svv,Suv,k1,k2)
    else:
        return(Suu,Svv,Suv,k1,k2)
 
# In[Ring average of spectra]
def spectra_average(S_image,k,bins,angle_bin = 30,stat=False):
    """
    S_r = spectra_average(S_image,k,bins)
    
    A function to reduce 2D Spectra to a radial cross-section.
    
    INPUT:
    ------
        S_image   - 2D Spectra array.
        
        k         - Tuple containing (k1_max,k2_max), wavenumber axis
                    limits
        bins      - Number of bins per decade.
        
        angle_bin - Sectors to determine spectra alignment
        
        stat      - Bin statistics output
        
     OUTPUT:
     -------
      S_r - a data structure containing the following
                   statistics, computed across each annulus:
          .k      - horizontal wavenumber k**2 = k1**2 + k2**2
          .S      - mean of the Spectra in the annulus
          .std    - std. dev. of S in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
#    import numpy as np

    class Spectra_r:
        """Empty object container.
        """
        def __init__(self): 
            self.S = None
            #self.std = None
            #self.median = None
            #self.numel = None
            #self.max = None
            #self.min = None
            #self.k1 = None
            #self.k2 = None
    
    #---------------------
    # Set up input parameters
    #---------------------
    S_image = np.array(S_image)
    npix, npiy = S_image.shape       
        
    k1 = k[0]#*np.linspace(-1,1,npix)
    k2 = k[1]#*np.linspace(-1,1,npiy)
    k1, k2 = np.meshgrid(k1,k2)
    # Polar coordiantes (complex phase space)
    r = np.absolute(k1+1j*k2)
    
    # Ordered 1 dimensinal arrays
    #ind = np.argsort(r.flatten())
    
#    r_sorted = r.flatten()[ind]
#    Si_sorted = S_image.flatten()[ind]
#    r_log = np.log(r_sorted)
#    r_log10 = np.log10(r_sorted)
#    decades = len(np.unique(r_log10.astype(int))) 
#    bin_tot = decades*bins
#    r_bin10 = np.linspace(np.min(r_log10.astype(int))-1,np.max(r_log10.astype(int)),bin_tot+1)
#    mat = np.array([r_log10/rb<1  for rb in r_bin10[1:]])
#    # bin number array
#    r_n_bin = np.sum(mat,axis=0)
#    # Find all pixels that fall within each radial bin.
#    delta_bin = r_n_bin[1:] - r_n_bin[:-1]
#    bin_ind = np.where(delta_bin)[0] # location of changes in bin
#    nr = bin_ind[1:] - bin_ind[:-1]  # number of elements per bin 
#    r_log = r_n_bin*np.max(r_log)/bin_tot
#    bin_centers= np.exp(0.5*(r_log[bin_ind[1:]]+r_log[bin_ind[:-1]]))
#    
#    # Cumulative sum to 2D spectra to find sum per bin
#    csSim = np.cumsum(Si_sorted, dtype=float)
#    tbin = csSim[bin_ind[1:]] - csSim[bin_ind[:-1]]
#    
    
#    r_sorted = r.flatten()#[ind]
    Si_sorted = S_image.flatten()#[ind]
#    r_log = np.log(r_sorted)
    r_log10 = np.log10(r.flatten())
    decades = len(np.unique(r_log10.astype(int)))
    
    bin_tot = decades*bins
    r_bin10 = np.linspace(np.min(r_log10.astype(int))-1,np.max(r_log10.astype(int)),bin_tot+1)
    
#    mat = np.array([r_log10/rb<1  for rb in r_bin10[1:]])
#    
#    
#    r_n_bin = np.sum(mat,axis=0)
#    
#    # Find all pixels that fall within each radial bin.
#    delta_bin = r_n_bin[1:] - r_n_bin[:-1]
#    
#    bin_ind = np.r_[0,np.where(delta_bin)[0]+1,len(r_n_bin)]# location of changes in bin
#    nr = bin_ind[1:] - bin_ind[:-1]
#    bin_ind = np.r_[np.where(delta_bin)[0],len(r_n_bin)-1]
#    csSim = np.cumsum(Si_sorted, dtype=float)
#    tbin = np.r_[csSim[bin_ind[0]],csSim[bin_ind[1:]] - csSim[bin_ind[:-1]]]
    
#    bin_centers = np.sqrt(10**r_bin10[r_n_bin[bin_ind]]*10**r_bin10[r_n_bin[bin_ind]+1])
    
    S_ring = np.zeros(Si_sorted.shape)#tbin/nr
    
#    rlog10 = r_bin10[r_n_bin[bin_ind]]
    
#    for i in range(len(rlog10)-1):
#        
#        ind0 = (r_log10>rlog10[i]) & (r_log10<rlog10[i+1])
#        print(i,tbin[i]/nr[i],np.sum(ind0),rlog10[i],rlog10[i+1])
#        S_ring[ind0] = tbin[i]/nr[i]
        
        
    for i in range(len(r_bin10)-1):
        ind0 = (r_log10>r_bin10[i]) & (r_log10<r_bin10[i+1])
        S_ring[ind0] = np.sum(S_image.flatten()[ind0])/np.sum(ind0)
    
#    plt.figure()
#    plt.plot(10**r_bin10[r_n_bin[bin_ind]],tbin/nr)
#    plt.xscale('log')    
#    plt.yscale('log')  
    
    
    
    
#    S_ring[ind] = S_ring
#    plt.figure()
#    plt.contourf(k1, k2,S_image)
#    plt.colorbar()
#    plt.figure()
#    plt.contourf(k1, k2,np.reshape(S_ring,S_image.shape))
#    plt.colorbar()
    
    
#    # Same for angles and orientation
#    k1 = k[0]*np.linspace(-1,1,npix)
#    k2 = k[1]*np.linspace(-1,1,npiy/2)
#    k1, k2 = np.meshgrid(k1,k2)
#    phi = np.angle(k1+1j*k2)
#
#    ind_p = np.argsort(phi.flatten())
#    
#    Si_sorted_p = S_image.flatten()[ind_p]
#    phi_sorted = phi.flatten()[ind_p]
#    phi_int = (phi_sorted*180/np.pi/angle_bin).astype(int)
#    delta_bin = phi_int[1:] - phi_int[:-1]
#    phiind = np.r_[np.where(delta_bin)[0],len(delta_bin)] # location of changes in sector
#    nphi = phiind[1:] - phiind[:-1] 
#    # The other half
#    # Cumulative sum to figure out sums for each radius bin
#    csSim_p = np.cumsum(Si_sorted_p, dtype=float)
#    tbin_p = csSim_p[phiind[1:]] - csSim_p[phiind[:-1]]

    # Initialization of the data
    S_r = Spectra_r()
    S_r.S = np.reshape(S_ring,S_image.shape)
#    S_r.S_p = tbin_p / nphi
#    S_r.k1 = k1
#    S_r.k2 = k2
#    S_r.phi = 0.5*(phi_int[phiind[1:]]+phi_int[phiind[:-1]])*angle_bin
    # optional?
    if stat==True:
        S_stat = np.array([[np.std(Si_sorted[r_n_bin==r]), 
                            np.median(Si_sorted[r_n_bin==r]),
                            np.max(Si_sorted[r_n_bin==r]),
                            np.min(Si_sorted[r_n_bin==r])]
                            for r in np.unique(r_n_bin)])        
        S_r.std = S_stat[:,0]
        S_r.median = S_stat[:,1]
        S_r.max = S_stat[:,2]
        S_r.min = S_stat[:,3]
    
    return S_r
# In[Just spectra]
def spectra_fft(tri,grid,U,V):
    """
    Function to estimate autocorrelation from a single increment in cartesian
    coordinates(tau, eta)
    Input:
    -----
        tri      - Delaunay triangulation object of unstructured grid.
                   
        grid     - Squared, structured grid to apply FFT.
        
        U, V     - Arrays with cartesian components of wind speed.
        
    Output:
    ------
        r_u,r_v  - 2D arrays with autocorrelation function rho(tau,eta) 
                   for U and V, respectively.               
    """
    U = U-np.nanmean(U)
    V = V-np.nanmean(V)
    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    
    n = grid[0].shape[0]
    m = grid[1].shape[0]
    
    U_int = np.reshape(LinearTriInterpolator(tri, U)(grid[0].flatten(),
                                grid[1].flatten()).data,grid[0].shape)    
    V_int = np.reshape(LinearTriInterpolator(tri, V)(grid[0].flatten(),
                                grid[1].flatten()).data,grid[1].shape)
    #zero padding
    U_int[np.isnan(U_int)] = 0.0
    V_int[np.isnan(V_int)] = 0.0
    fftU = np.fft.fftshift(np.fft.fft2(U_int))
    fftV = np.fft.fftshift(np.fft.fft2(V_int))
    fftUV = fftU*np.conj(fftV)
    Suu = 2*(np.abs(fftU)**2)/(n*m*dx*dy)
    Svv = 2*(np.abs(fftV)**2)/(n*m*dx*dy)
    Suv = 2*np.real(fftUV)/(n*m*dx*dy) 
    kx = 1/(2*dx)
    ky = 1/(2*dy) 
    k1 = kx*np.linspace(-1,1,len(Suu))
    k2 = ky*np.linspace(-1,1,len(Suu))
    return(Suu,Svv,Suv,k1,k2)
    
# In[]
    
def avetriangles(xy,z,tri):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        triangles: ntri, dim+1 indices of triangles or simplexes, as from
                   http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        areasum: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
    # z concave or convex => under or overestimates
    ind = ~np.isnan(z)
    xy = xy[ind,:]
    z = z[ind]
    triangles = Triangulation(xy[:,0],xy[:,1]).triangles
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    zsum = np.zeros( z[0].shape )
    areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for tri in triangles:
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area = abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area = abs( np.linalg.det( t )) / dimfac  # v slow
        aux = area * np.nanmean(z[tri],axis=0)
        if ~np.isnan(aux):
            zsum += aux
            areasum += area
    return zsum/areasum

# In[]   
def upsample2 (x, k):
  """
  Upsample the signal to the new points using a sinc kernel. The
  interpolation is done using a matrix multiplication.
  Requires a lot of memory, but is fast.
  input:
  xt    time points x is defined on
  x     input signal column vector or matrix, with a signal in each row
  xp    points to evaluate the new signal on
  output:
  y     the interpolated signal at points xp
  """

  mn = x.shape

  if len(mn) == 2:
    m = mn[0]
    n = mn[1]
  elif len(mn) == 1:
    m = 1
    n = mn[0]
  else:
    raise ValueError ("x is greater than 2D")

  nn = n * k

  [T, Ts]  = np.mgrid[1:n:nn*1j, 1:n:n*1j]
  TT = Ts - T
  del T, Ts

  y = np.sinc(TT).dot (x.reshape(n, 1))

  return y.squeeze()  

# In[Lanczos polar]
  
def lanczos_kernel(r,r_1 = 1.22,r_2 = 2.233,a=1):
    kernel = lambda r: 2*sp.special.jv(1,a*np.pi*r)/r/np.pi/a
    kernel_w = kernel(r)*kernel(r*r_1/r_2)
    kernel_w[np.abs(r)>=r_2] = 0.0
    return kernel_w
    
def lanczos_int_sq(grid,tree,U,a=1):    
    dx = np.max(np.diff(grid[0].flatten()))
    dy = np.max(np.diff(grid[1].flatten()))
    X = grid[0].flatten()
    Y = grid[1].flatten()
    tree_grid = KDTree(np.c_[X,Y])
    d, n  = tree.query(tree_grid.data, k=40, return_distance = True)
    d=d/np.sqrt(dx*dy)
    S = np.sum(lanczos_kernel(d)*U[n],axis=1)
    S = np.reshape(S,grid[0].shape)
    return S

# In[]
#def spatial_spec_sq(x,y,U,V,transform = False):
#    
#    
#    if transform:
#        U_mean = np.nanmean(U.flatten())
#        V_mean = np.nanmean(V.flatten())
#        # Wind direction
#        gamma = np.arctan2(V_mean,U_mean)
#        # Components in matrix of coefficients
#        S11 = np.cos(gamma)
#        S12 = np.sin(gamma)
#        T = np.array([[S11,S12], [-S12,S11]])
#        vel = np.array(np.c_[U.flatten(),V.flatten()]).T
#        vel = np.dot(T,vel)
#        X = np.array(np.c_[x,y]).T
#        X = np.dot(T,X)   
#        U_t = np.reshape(vel[0,:],U.shape).T
#        V_t = np.reshape(vel[1,:],V.shape).T       
#        U_mean = np.nanmean(U_t.flatten())
#        V_mean = np.nanmean(V_t.flatten())        
#        U_t = U_t-U_mean
#        V_t = V_t-V_mean
#        x = X[0,:]
#        y = X[1,:]
#    else:
#        U_mean = np.nanmean(U.flatten())
#        V_mean = np.nanmean(V.flatten())
#        U_t = U-U_mean
#        V_t = V-V_mean        
#    dx = np.min(np.abs(np.diff(x)))
#    dy = np.min(np.abs(np.diff(y)))
#    grid = np.meshgrid(x,y)
#    U_t[np.isnan(U_t)] = 0.0
#    V_t[np.isnan(V_t)] = 0.0
#    n = grid[0].shape[0]
#    m = grid[1].shape[0]
#    # Spectra
#    fftU = np.fft.fft2(U_t)
#    fftV = np.fft.fft2(V_t)
#    fftU  = np.fft.fftshift(fftU)
#    fftV  = np.fft.fftshift(fftV) 
#    fftUV = fftU*np.conj(fftV) 
#    Suu = 2*(np.abs(fftU)**2)*dx*dy/(n*m)
#    Svv = 2*(np.abs(fftV)**2)*dx*dy/(n*m) 
#    Suv = 2*np.real(fftUV)*(dx*dy)/(n*m)
#    k1 = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
#    k2 = np.fft.fftshift((np.fft.fftfreq(m, d=dy)))
#    print(X.shape,n,m,k1.shape,k2.shape)
#    Su_u = sp.integrate.simps(Suu,k2,axis=1)[k1>0]
#    Sv_v = sp.integrate.simps(Svv,k2,axis=1)[k1>0]
#    Su_v = sp.integrate.simps(Suv,k2,axis=1)[k1>0]
#    k1 = k1[k1>0]
#    F = .5*(Su_u + Sv_v)  
#    
#    return (k1,Su_u,Sv_v,Su_v,F,x,y)
# In[]

def spatial_spec_sq(x,y,U,V,transform = False,shrink = False, ring=False,bins=20):
    
    grid = np.meshgrid(x,y)
    dx = np.min(np.abs(np.diff(x)))
    dy = np.min(np.abs(np.diff(y))) 
    # Shrink and square
    if shrink:
        patch = ~np.isnan(U)
        ind_patch_x = np.sum(patch,axis=1) != 0
        ind_patch_y = np.sum(patch,axis=0) != 0
        if np.sum(ind_patch_x) > np.sum(ind_patch_y):
            ind_patch_y = ind_patch_x
        elif np.sum(ind_patch_y) > np.sum(ind_patch_x):
            ind_patch_x = ind_patch_y        
        n = np.sum(ind_patch_x)
        m = np.sum(ind_patch_y)          
        ind_patch_grd = np.meshgrid(ind_patch_y,ind_patch_x)
        ind_patch_grd = ind_patch_grd[0] & ind_patch_grd[1]
        U = np.reshape(U[ind_patch_grd],(n,m))
        V = np.reshape(V[ind_patch_grd],(n,m))
        grid[0] = np.reshape(grid[0][ind_patch_grd],(n,m))
        grid[1] = np.reshape(grid[1][ind_patch_grd],(n,m))        
    else:
        n = grid[0].shape[0]
        m = grid[1].shape[0]   
    k1 = np.fft.fftshift((np.fft.fftfreq(n, d=dx)))
    k2 = np.fft.fftshift((np.fft.fftfreq(m, d=dy))) 
    k1, k2 = np.meshgrid(k1,k2)
    k1p, k2p = k1, k2
    
    if transform:
        U_mean = np.nanmean(U.flatten())
        V_mean = np.nanmean(V.flatten())
        # Wind direction
        gamma = np.arctan2(V_mean,U_mean)
        # Components in matrix of coefficients
        S11 = np.cos(gamma)
        S12 = np.sin(gamma)
        T = np.array([[S11,S12], [-S12,S11]])
        
        K = np.array(np.c_[k1.flatten(),k2.flatten()]).T
        K = np.dot(T,K) 
        k1 = np.reshape(K[0,:],k1.shape)
        k2 = np.reshape(K[1,:],k2.shape)
    U_mean = np.nanmean(U.flatten())
    V_mean = np.nanmean(V.flatten())
    U_t = U-U_mean
    V_t = V-V_mean
    
    U_t[np.isnan(U_t)] = 0.0
    V_t[np.isnan(V_t)] = 0.0
    # Spectra
    fftU = np.fft.fft2(U_t)
    fftV = np.fft.fft2(V_t)
    fftU  = np.fft.fftshift(fftU)
    fftV  = np.fft.fftshift(fftV) 
    fftUV = fftU*np.conj(fftV) 
    
    Suu = 2*(np.abs(fftU)**2)*dx*dy/(n*m)
    Svv = 2*(np.abs(fftV)**2)*dx*dy/(n*m) 
    Suv = 2*np.real(fftUV)*(dx*dy)/(n*m)
    
    dk = np.min([np.max(k1p.flatten()),np.max(k2p.flatten())])/np.sqrt(2)
    
    k1_int = np.linspace(-dk,dk,512)
    k2_int = k1_int
    k_int_grd = np.meshgrid(k1_int,k2_int)
    
    mask = (k1<dk*1.1) & (k1>-dk*1.1)
    
    print(len(k1[mask]),len(k1.flatten()))
    
    Suu_int = sp.interpolate.griddata(np.c_[k1[mask],k2[mask]],
              Suu[mask], (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
              method='nearest')    
    Svv_int = sp.interpolate.griddata(np.c_[k1[mask],k2[mask]],
          Svv[mask], (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
          method='nearest')    
    Suv_int = sp.interpolate.griddata(np.c_[k1[mask],k2[mask]],
          Suv[mask], (k_int_grd[0].flatten(),k_int_grd[1].flatten()),
          method='nearest')
    
    Suu_int = np.reshape(Suu_int,k_int_grd[0].shape)
    Svv_int = np.reshape(Svv_int,k_int_grd[0].shape)
    Suv_int = np.reshape(Suv_int,k_int_grd[0].shape)
    
#    fig, ax = plt.subplots()
#    im=ax.contourf(k1_int,k2_int,np.log10(Suu_int),np.linspace(0,7,300),cmap='jet')
#    ax.set_xlabel('$k_1$', fontsize=18)
#    ax.set_ylabel('$k_2$', fontsize=18)
#    ax.set_xlim(-.005,0.005)
#    ax.set_ylim(-.005,0.005)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    cbar = fig.colorbar(im, cax=cax)
#    cbar.ax.tick_params(labelsize=18)
#    ax.tick_params(labelsize=18)
#    cbar.ax.set_ylabel("$\log_{10}{S_{uu}}$", fontsize=18)
#    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=18,verticalalignment='top')
    
    print(2)


#    l = np.linspace(15,2*10**5,100)
#    print(gamma*180/np.pi)
#    
#    fig, ax = plt.subplots()
#    ax.set_aspect('equal') 
#    ax.contourf(k1,k2,Suu,l,locator=ticker.LogLocator(),cmap='jet')
#    
#    fig, ax = plt.subplots()
#    ax.set_aspect('equal') 
#    ax.contourf(k1_int,k2_int,Suu_int,l,locator=ticker.LogLocator(),cmap='jet')
#    
#    fig, ax = plt.subplots()
#    ax.set_aspect('equal') 
#    ax.contourf(k1p,k2p,Suu,l,locator = ticker.LogLocator(),cmap='jet')
    if ring:
        Su=spectra_average(Suu_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #ku = Su.k        
        Suu_int = Su.S    
        Sv=spectra_average(Svv_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #kv = Sv.k
        Svv_int = Sv.S
        Suv=spectra_average(Suv_int,(k1_int, k2_int),bins,angle_bin = 30,stat=False)
        #kuv = Suv.k
        Suv_int = Suv.S
        #return (ku,kv,kuv,Su_u,Sv_v,Su_v)
#    else:
#    fig, ax = plt.subplots()
#    im=ax.contourf(k1_int,k2_int,np.log10(Suu_int),np.linspace(0,7,300),cmap='jet')
#    ax.set_xlabel('$k_1$', fontsize=18)
#    ax.set_ylabel('$k_2$', fontsize=18)
#    ax.set_xlim(-.005,0.005)
#    ax.set_ylim(-.005,0.005)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    cbar = fig.colorbar(im, cax=cax)
#    cbar.ax.tick_params(labelsize=18)
#    ax.tick_params(labelsize=18)
#    cbar.ax.set_ylabel("$\log_{10}{S_{uu}}$", fontsize=18)
#    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=18,verticalalignment='top')

    
    Su_u = sp.integrate.simps(Suu_int,k2_int,axis=1)[k1_int>0]
    Sv_v = sp.integrate.simps(Svv_int,k2_int,axis=1)[k1_int>0]
    Su_v = sp.integrate.simps(Suv_int,k2_int,axis=1)[k1_int>0]
    return (k1_int[k1_int>0],k2_int[k1_int>0],Su_u,Sv_v,Su_v)      
    
    
    
 
