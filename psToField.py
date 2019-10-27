import numpy as np 
from numpy.fft import fftn, fftshift 

def fft_return_real(ndims,n):
    half = n//2

    if ndims == 3:
        field = np.zeros((n,n,n), dtype = complex)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    field[i,j,k] = np.random.normal() + 1j*np.random.normal()
                    reflection = (-1*( np.array([i,j,k]) - half) + half)%n
                    field[tuple(reflection.astype(int))] = np.conj(field[i,j,k])
        field[half,half,half] = 1
        field[0,0,half] = 1
        field[0,half,0] = 1
        field[0,0,half] = 1
        field[0,0,0] = 1

    elif ndims ==2:
        field = np.zeros((n,n), dtype = complex)
        for i in range(n):
            for j in range(n):
                field[i,j] = np.random.normal() + 1j*np.random.normal()
                reflection = (-1*( np.array([i,j]) - half) + half)%n
                field[tuple(reflection.astype(int))] = np.conj(field[i,j])

        field[half,half] = 1
        field[0,half] = 1
        field[half,0] = 1
        field[0,0] = 1

    return field

def r3_norm(rx,ry,rz):
    '''calculating length of each voxel's radial distance from origin'''
    return np.sqrt(rx**2 + ry**2 + rz**2)

def radii(ndims,n,L):
    
    origin = n//2
    delta_k = 2*np.pi/L
    if ndims == 2: 
        x,y = np.indices((n,n))
        radii = np.hypot(x - origin, y - origin)*delta_k
        
    elif ndims == 3: 
        x,y,z = np.indices((n,n,n))
        radii = r3_norm(x-origin, y - origin, z - origin)*delta_k
    
    return radii

def field(ps_function, shape, L):
    '''returns a field in configuration space given a function describing the 
    power spectrum of the field
    -ps_function: function defining power spectrum (one paramater k)
    -shape: shape of desired field. Tuple, either (n,n,n) or (n,n)
    -L: realspace length of one side of the box in Mpc
    '''
    ndims = len(shape)
    n = shape[0]

    #complex random gauss grid satisfying Im(fft(rand_gauss)) ~ 0
    rand_gauss = fft_return_real(ndims,n)

    #distance from origin of each pixel/voxel
    r = radii(ndims,n,L)
    
    #each pixel will be drawn from gaussian distribution of this stdv
    stdv_r = np.sqrt( (0.5)*(n**2)*ps_function(r) )

    #scaling random gaussian dist according to given power spectrum
    kspace = rand_gauss * stdv_r 
    return np.real(fftshift(fftn(fftshift(kspace))))
