import numpy as np
from numpy.fft import fft2, fftn, fftshift

'''This function returns the power spectrum of a given grid/box of pixels

Current Version: May 10 2019
'''

def combine_bins(bins, dims, n):
    '''given some bins and their dimensions, combines the first n bins. Also
    returns new dimension array.

    -bins: array of binned pixels
    -dims: array containing info about number of pixels per bins
    -n: number of bins to combine'''

    b = np.concatenate(([sum(bins[:n])], bins[n:]))
    c = np.concatenate(([sum(dims[:n])],dims[n:]))
    return(b, c)

def group(array, group_indices):
    '''grouping an array into bins. This is the problematic/slow
    part of the code I think? Gotta be a better way to do it

    -array: data to bin
    -group_indices: array where group_indices[i] gives the index in
    the array of the first element going into the ith bin.'''
    grouped = []
    for i in range(1, len(group_indices)):
        grouped.append(array[group_indices[i-1]:group_indices[i]]  )
    return np.array(grouped)

def grid2d(data):
    '''returns grid of pixel positions, origin, and max radius (2d grid)
     -r_max: since our origin is not geometric center of grid, this is the
     radius of the circle/sphere that reaches the edge of the grid, centered on 
     the fft origin.
    '''
    n = data.shape[0]
    y,x = np.indices(data.shape)
    origin = [n//2,n//2]
    r_max = (n-0.5) - origin[0]

    return (x,y,origin, r_max, n)

def grid3d(data):
    '''returns grid of pixel positions, origin, and max radius (3d grid)
     -r_max: since our origin is not geometric center of grid, this is the
     radius of the circle/sphere that reaches the edge of the grid, centered on 
     the fft origin.
    '''
    n = data.shape[0]
    x,y,z = np.indices(data.shape)
    origin = [n//2,n//2,n//2]
    r_max = (n-0.5) - origin[0]

    return (x,y,z, origin, r_max, n)

def r3_norm(rx,ry,rz):
    '''calculating length of each voxel's radial distance from origin'''
    return np.sqrt(rx**2 + ry**2 + rz**2)

def radial_distances2d(kspace_abs):
    x,y,origin,r_max,n = grid2d(kspace_abs)
    radii = np.hypot(x - origin[0], y - origin[1])
    return (radii, r_max, n)

def radial_distances3d(kspace_abs):
    x,y,z,origin,r_max,n = grid3d(kspace_abs)
    radii = r3_norm(x-origin[0], y - origin[1], z - origin[2])
    return (radii, r_max, n)

def fourier_space(data, ndims, resolution):
    data_shift = fftshift(data)
    if ndims is 3: 
        kspace = np.abs(fftshift(fftn(data_shift)))**2
        r, r_max, n = radial_distances3d(kspace)
        survey_size = (n*resolution)**3
    elif ndims is 2: 
        kspace = np.abs(fftshift(fft2(data_shift)))**2
        r, r_max, n = radial_distances2d(kspace)
        survey_size = (n*resolution)**2

    resolution_k = 1/(n*resolution)
    return r, r_max, kspace, survey_size, resolution_k

def standard_error(bin_array):
    '''computes standard error'''
    standard_errors = []
    for arr in bin_array:
        standard_errors.append(np.std(arr))
    return standard_errors

def p_spec(data, resolution = 1, ndims = 2, n_bins =None, bin_w = None, combine = None):
    ''' Returns: kmodes, power, err_power, err_k.
    Parameters:
    -data: configuration space grid/box
    -resolution: the length scale corresponding to one pixel/voxel
    -ndims: int, 2 or 3. dimensions of data
    -n_bins:  number of radial bins desired
    -bin_w: alternative to n_bins, if want to specify the width of bins instead
    -combine: int, number of bins to combine if want to use the combine_bins function
    '''
    #radial distances from origin of each pixel/voxel, max radius to consider, 
    #kspace data, survey area/volume, and kspace resolution. 
    r, r_max, k_data, survey_size, resolution_k = fourier_space(data, ndims, resolution) 
    
    if bin_w is None:
        bin_w = r_max/n_bins #thickness of each radial bin

    bins = np.arange(0,r_max+bin_w,bin_w) #bin ranges

    ind = np.argsort(r.flat)
    #ind is here so as to not lose track of which pixel corresponds to which
    # radius after sorting
    r_sorted = r.flat[ind]
    pix_sorted = k_data.flat[ind]

    #indices corresponding to first element in each bin
    indices = np.array([np.searchsorted(r_sorted, bins) for bins in bins])
    bin_dims = indices[1:] - indices[:-1] # num of pixels per bin

    #pixels and corresponding radii binned
    pix_binned = group(pix_sorted, indices)
    r_binned = group(r_sorted, indices)

    bin_sum = [np.sum(bins) for bins in pix_binned] #optmize this also?
    r_sum = [np.sum(bins) for bins in r_binned]


    if combine is not None:
        bin_sum, bin_dims = combine_bins(bin_sum,bin_dims, combine)
        r_sum, bin_dims = combine_bins(r_sum,bin_dims, combine)

    power = np.array(bin_sum/bin_dims)/survey_size
    kmodes = np.array(r_sum/bin_dims)*resolution_k*2*np.pi

    err_power = standard_error(pix_binned)/survey_size
    err_k = standard_error(r_binned)*resolution_k*2*np.pi

    return kmodes, power, err_power, err_k
