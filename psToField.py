import numpy as np 
from numpy.fft import fft2, fftshift 
from fftReturnReal import fft_return_real 


def radii(n):
    '''returns the radial distance of each pixel in n by n grid from center.'''
    y, x = np.indices((n,n))
    center = [n//2,n//2]
    return np.hypot(x - center[0], y - center[1])

def field(ps_function, n = 1000):
    '''returns a field in configuration space given a function describing the 
    power spectrum of the field
    -ps_function: function defining power spectrum (one paramater k)
    -n: grid size
    '''
    
    #complex random gauss grid satisfying Im(fft2(rand_gauss)) ~ 0
    rand_gauss = fft_return_real(n)
    r = radii(n)
    
    #each pixel will be drawn from gaussian distribution of this stdv
    stdv_r = np.sqrt( (0.5)*(n**2)*ps_function(r) )

    #scaling random gaussian dist according to given power spectrum
    kspace = rand_gauss * stdv_r 
    return np.real(fftshift(fft2(fftshift(kspace))))
