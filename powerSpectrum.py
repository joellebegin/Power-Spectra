import numpy as np
from numpy.fft import fftn, fftshift

def fourier(f):
    return fftshift(fftn(fftshift(f)))

class PowerSpectrum():
    def __init__(self, field, bins = None, bin_w = 1, L = 300):
        '''
        - field: field in fourier space
        - bins: array, if want to give bin edges in fourier space units
        - bin_w: else, specify bin width in pixel units
        - L: real space length of box
        '''
        self.field = field
        self.bins = bins
        self.bin_w = bin_w
        self.L = L
        
        self.ndims = len(field.shape)
        self.n = field.shape[0] #number of pixels along one axis
        self.abs_squared = np.abs(field)**2 #amplitude of field squared
        self.delta_k = 2*np.pi/self.L #kspace resolution of 1 pixel
        self.survey_size = (L**self.ndims)
    
    def r3_norm(self,rx,ry,rz):
        '''calculating length of each voxel's radial distance from origin'''
        return np.sqrt(rx**2 + ry**2 + rz**2)
    
    def grid(self):
        '''defines some useful variables
        -r_max: maximum radial distance we consider
        - radii: grid that contains radial distance of each pixel from origin, 
        in pixel units'''
        
        origin = self.n//2
        self.rmax = self.n - 0.5 - origin
        
        if self.ndims == 2: 
            x,y = np.indices(self.field.shape)
            self.radii = np.hypot(x - origin, y - origin)*self.delta_k
            
        elif self.ndims == 3: 
            x,y,z = np.indices(self.field.shape)
            self.radii = self.r3_norm(x-origin, y - origin, z - origin)*self.delta_k
    
    def sort(self):
        sort_ind = np.argsort(self.radii.flat)
        self.r_sorted = self.radii.flat[sort_ind]
        self.vals_sorted = self.abs_squared.flat[sort_ind] 
        
    def get_bin_ind(self):
        bin_ind = [1]
        for bin_val in self.bins:
            val = np.argmax( self.r_sorted > bin_val)
            if val == 0: #ie bin_val > r_max
                val = len(self.r_sorted)
            bin_ind.append(val-1)
        self.bin_ind = np.array(bin_ind)

    def average_bins(self):
        vals_binned = []
        r_binned = []
        bin_dims = []
        
        for i in range(1, len(self.bin_ind)):
            r_binned.append(np.sum(self.r_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1]))
            vals_binned.append(np.sum(self.vals_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1]))
            bin_dims.append(len(self.r_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1]))
        
        self.vals_binned = np.array(vals_binned)
        self.r_binned = np.array(r_binned)
        self.bin_dims = np.array(bin_dims)
        
        self.field_bins = self.vals_binned/self.bin_dims
        self.average_k = self.r_binned/self.bin_dims
    
    def p_spec(self):
        
        self.grid() #sets up grid of radial distances
        
        if self.bins is None: 
            #use bin_w sparingly. I don't trust it
            self.bins = np.arange(0,self.rmax + self.bin_w, self.bin_w)*self.delta_k
        
        self.sort() #sorting radii + field vals (increasing)
        self.get_bin_ind() #indices determing which elements go into bins
        self.average_bins() #computing average of bins
        self.power = self.field_bins/self.survey_size
        
    def compute_pspec(self, del_squared = True, avg_k = True):
        self.p_spec()
        
        delta_squared = self.average_k**3*self.power/(2*np.pi**2)
        return self.average_k, delta_squared

def main():
    filename = input()

    data = np.loadtxt(filename, delimiter=',')
    box = np.reshape(data, (200,200,200))

    bins =np.logspace(np.log10(0.021), np.log10(3), num=13)

    k,delk = PowerSpectrum(fourier(box), bins = bins).compute_pspec()
    
    np.savetxt( filename + '.csv', np.vstack((k,delk)), delimiter=',')

if __name__ == "__main__":
    main()