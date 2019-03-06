import numpy as np
from numpy.fft import fft2, fftshift

'''given some bins and their dimensions, combines the first n bins (also returns new dimension array)'''
def combine_bins(bins, dims, n):
    b = np.concatenate(([sum(bins[:n])], bins[n:]))
    c = np.concatenate(([sum(dims[:n])],dims[n:]))
    return(b, c)

'''grouping pixels into their respective bins. This is the problematic/slow part of the code I think?
Gotta be a better way to do it'''
def group_pixels(sorted_pix, bin_indices):
    grouped_pix = [] 
    for i in range(1, len(bin_indices)):
        grouped_pix.append(sorted_pix[bin_indices[i-1]:bin_indices[i]]  )
    return np.array(grouped_pix)
    
"creates grid of pixel positions, the center of the grid, and max radius"
def grid(data):
    y,x = np.indices(data.shape)
    center = [n//2,n//2]
    r_max = (n-0.5) - center[0]
    
    return (x,y,center, r_max)

def power(data, n_bins =None, combine = None, bin_w = None):
    data = fftshift(data)
    kspace = np.abs(fftshift(fft2(data)))**2

    x,y,center,r_max = grid(kspace)
    
    r = np.hypot(x - center[0], y - center[1]) #radial distance of each pixel from center
    if bin_w is None: 
        bin_w = r_max/n_bins #thickness of each radial bin
    
    bins = np.arange(0,r_max+bin_w,bin_w) #bin ranges
    
    ind = np.argsort(r.flat)#sorting radii and their corresponding pixels
    r_sorted = r.flat[ind]
    pix_sorted = kspace.flat[ind]
    
    #indices corresponding to first element in each bin
    indices = np.array([np.searchsorted(r_sorted, bins) for bins in bins]) #can probably optimize
    bin_dims = indices[1:] - indices[:-1] # num of pixels per bin
    
    bins = group_pixels(pix_sorted, indices)
    bin_sum = [np.sum(bins) for bins in bins] #optmize this also?
    
    if combine is not None: 
        bin_sum, bin_dims = combine_bins(bin_sum,bin_dims, combine) 
        
    return np.array(bin_sum/bin_dims)/(n**2)