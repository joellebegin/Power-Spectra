import numpy as np 
from numpy.fft import fft2, fftshift 
from fftReturnReal import real 


'''returns the radial distance of each pixel in n by n grid from center.'''
def radii(n):
    y, x = np.indices((n,n))
    center = [n//2,n//2]
    return np.hypot(x - center[0], y - center[1])

def field(ps_function, n = None):
    
    #arbitrary grid size. When units update comes, this will be different.
    if n is None:
        n = 200 
    
    #complex random gauss grid satisfying Im(fft(randgauss)) = 0
    rand_gauss = real(n)
    r = radii(n)
    stdv_r = np.sqrt( (n**2)*ps_function(r) )

    #scaling random gaussian dist according to given power spectrum
    kspace = rand_gauss * stdv_r 
    return np.real(fftshift(fft2(fftshift(kspace))))