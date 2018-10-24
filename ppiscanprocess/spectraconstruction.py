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
from matplotlib.tri import Triangulation,UniformTriRefiner,CubicTriInterpolator,LinearTriInterpolator,TriFinder,TriAnalyzer
                           
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
    Suu = 2*(np.abs(fftU)**2)/(n*m*dx*dy)
    Svv = 2*(np.abs(fftV)**2)/(n*m*dx*dy)
    Suv = 2*np.real(fftUV)/(n*m*dx*dy)
    kx = 1/(2*dx)
    ky = 1/(2*dy)   
    k1 = kx*np.linspace(-1,1,len(Suu))
    k2 = ky*np.linspace(-1,1,len(Suu))
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
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.k = None
    
    #---------------------
    # Set up input parameters
    #---------------------
    S_image = np.array(S_image)
    npix, npiy = S_image.shape       
        
    k1 = k[0]*np.linspace(-1,1,npix)
    k2 = k[1]*np.linspace(-1,1,npiy)
    k1, k2 = np.meshgrid(k1,k2)
    # Polar coordiantes (complex phase space)
    r = np.absolute(k1+1j*k2)
    
    # Ordered 1 dimensinal arrays
    ind = np.argsort(r.flatten())
    
    r_sorted = r.flatten()[ind]
    Si_sorted = S_image.flatten()[ind]
    r_log = np.log(r_sorted)
    decades = len(np.unique(r_log.astype(int)))-1
    # Total number of bins
    bin_tot = decades*bins
    # bin number array
    r_n_bin = (r_log*bin_tot/np.max(r_log)).astype(int)
    
    # Find all pixels that fall within each radial bin.
    delta_bin = r_n_bin[1:] - r_n_bin[:-1]
    bin_ind = np.where(delta_bin)[0] # location of changes in bin
    nr = bin_ind[1:] - bin_ind[:-1]  # number of elements per bin 
    r_log = r_n_bin*np.max(r_log)/bin_tot
    bin_centers= np.exp(0.5*(r_log[bin_ind[1:]]+r_log[bin_ind[:-1]]))
    
    # Cumulative sum to 2D spectra to find sum per bin
    csSim = np.cumsum(Si_sorted, dtype=float)
    tbin = csSim[bin_ind[1:]] - csSim[bin_ind[:-1]]
    
    # Same for angles and orientation
    k1 = k[0]*np.linspace(-1,1,npix)
    k2 = k[1]*np.linspace(-1,1,npiy/2)
    k1, k2 = np.meshgrid(k1,k2)
    phi = np.angle(k1+1j*k2)

    ind_p = np.argsort(phi.flatten())
    
    Si_sorted_p = S_image.flatten()[ind_p]
    phi_sorted = phi.flatten()[ind_p]
    phi_int = (phi_sorted*180/np.pi/angle_bin).astype(int)
    delta_bin = phi_int[1:] - phi_int[:-1]
    phiind = np.r_[np.where(delta_bin)[0],len(delta_bin)] # location of changes in sector
    nphi = phiind[1:] - phiind[:-1] 
    # The other half
    # Cumulative sum to figure out sums for each radius bin
    csSim_p = np.cumsum(Si_sorted_p, dtype=float)
    tbin_p = csSim_p[phiind[1:]] - csSim_p[phiind[:-1]]

    # Initialization of the data
    S_r = Spectra_r()
    S_r.S = tbin / nr
    S_r.S_p = tbin_p / nphi
    S_r.k = bin_centers
    S_r.phi = 0.5*(phi_int[phiind[1:]]+phi_int[phiind[:-1]])*angle_bin
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
    
  
    
    
    
 
