import numpy as np
from numpy.fft import fft2, fftshift

'''This function returns the power spectrum of a given n by n grid of pixels
Current Version: May 2 2019
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
    
def grid(data):
    "returns grid of pixel positions, the center of the grid, and max radius"
    n = data.shape[0]
    y,x = np.indices(data.shape)
    center = [n//2,n//2]
    r_max = (n-0.5) - center[0]
    
    return (x,y,center, r_max, n)

def p_spec(data, resolution, n_bins =None, bin_w = None, combine = None):
    ''' Returns the power spectrum of a given (2d) grid in configuration space.

    -data: configuration space grid 
    -resolution: the length scale corresponding to one pixel
    -n_bins:  number of radial bins desired
    -bin_w: alternative to n_bins, if want to specify the width of bins instead
    -combine: int, number of bins to combine if want to use the combine_bins function 
    '''

    data = fftshift(data)
    kspace = np.abs(fftshift(fft2(data)))**2

    x,y,center,r_max,n = grid(kspace)
    resolution_k = 1/(n*resolution)
    area = (n*resolution)**2
    
    #r - radial distance of each pixel from center
    r = np.hypot(x - center[0], y - center[1]) 
    if bin_w is None: 
        bin_w = r_max/n_bins #thickness of each radial bin
    
    bins = np.arange(0,r_max+bin_w,bin_w) #bin ranges
    
    ind = np.argsort(r.flat)
    #ind is here so as to not lose track of which pixel corresponds to which 
    # radius after sorting
    r_sorted = r.flat[ind]
    pix_sorted = kspace.flat[ind]
    
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

    power = np.array(bin_sum/bin_dims)/area
    kmodes = np.array(r_sum/bin_dims)*resolution_k*2*np.pi
    return kmodes, power