import numpy as np
from numpy.fft import fftn, fftshift

class PowerSpectrum():

    def fourier(self,f):
        fourier_transform =  fftshift(fftn(fftshift(f)))
        scaled = fourier_transform*(self.delta_r**self.ndims)
        return scaled

    def __init__(self, field, bins = None, L = 300):
        '''
        - field: field in fourier space
        - bins: array, if want to give bin edges in fourier space units
        - bin_w: else, specify bin width in pixel units
        - L: real space length of box
        '''
        self.field = field
        self.field_fourier = self.fourier(field)

        if bins is not None:
            self.bins = bins
        else:
            self.bins = np.logspace(np.log10(0.021), np.log10(3), num=13)

        self.L = L
        
        self.ndims = len(field.shape)
        self.n = field.shape[0] #number of pixels along one axis
        self.abs_squared = np.abs(self.field_fourier)**2 #amplitude of field squared
        self.delta_k = 2*np.pi/self.L #kspace resolution of 1 pixel
        self.delta_r = self.L/self.n #real space resolution of 1 pixel

        self.survey_size = (self.n**self.ndims)#volume of box
    
    def r3_norm(self,rx,ry,rz):
        '''calculating length of each voxel's radial distance from origin'''
        return np.sqrt(rx**2 + ry**2 + rz**2)
    
    def grid(self, del_squared):
        '''defines some useful variables
        -r_max: maximum radial distance we consider
        -radii: grid that contains radial distance of each pixel from origin, 
        in kspace units'''
        
        origin = self.n//2
        self.rmax = self.n - 0.5 - origin
        if self.ndims == 2: 
            x,y = np.indices(self.field.shape)
            self.radii = np.hypot(x - origin, y - origin)*self.delta_k
            
        elif self.ndims == 3: 
            x,y,z = np.indices(self.field.shape)
            self.radii = self.r3_norm(x-origin, y-origin, z-origin)*self.delta_k
        
        if del_squared:
                self.abs_squared *= self.radii**3
                
    def sort(self):
        ''' sorts radii, and the field value corresponding to each radius.
        sort_ind is here so as to not lose track of which radius corresponds
        to which field value
        
        data is flattened in this step'''
        sort_ind = np.argsort(self.radii.flat)
        self.r_sorted = self.radii.flat[sort_ind]
        self.vals_sorted = self.abs_squared.flat[sort_ind] 

        if self.ignore_0:
            self.r_sorted = self.r_sorted[1:]
            self.vals_sorted = self.vals_sorted[1:]
        
    def get_bin_ind(self):
        '''given the desired bin edges, determines the index of the last pixel
        going into this bin'''
        bin_ind = [0]
        for bin_val in self.bins:
            val = np.argmax( self.r_sorted > bin_val)
            if val == 0: #ie bin_val > r_max
                val = len(self.r_sorted)
            bin_ind.append(val-1)
        self.bin_ind = np.array(bin_ind)

    def average_bins(self):
        ''' puts things in bins, averages the bins
        -average_k: average k value going into each bin
        -field_bins: field values put into bins and averaged'''
        vals_binned = []
        r_binned = []
        bin_dims = []
        
        for i in range(1, len(self.bin_ind)): #THIS IS SLOW AND UGLY. FIX ONE DAY
            r_binned.append(np.sum(self.r_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1]))
            vals_binned.append(np.sum(self.vals_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1]))
            bin_dims.append(len(self.r_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1]))
        
        self.vals_binned = np.array(vals_binned)
        self.r_binned = np.array(r_binned)
        self.bin_dims = np.array(bin_dims)
        
        self.field_bins = self.vals_binned/self.bin_dims
        self.average_k = self.r_binned/self.bin_dims
    
    def p_spec(self, del_squared, ignore_0):
        '''like the main method. Organizes stuff'''

        self.ignore_0 = ignore_0
        
        self.grid(del_squared) #sets up grid of radial distances
        
    
        self.sort() #sorting radii + field vals (increasing)
        self.get_bin_ind() #indices determing which elements go into bins
        self.average_bins() #computing average of bins
        self.power = self.field_bins/self.survey_size
        
        if del_squared:
            self.power *= (1/(2*np.pi**2))
        
    def compute_pspec(self, del_squared = True, ignore_0 = False):
        self.p_spec(del_squared, ignore_0)
        return self.average_k, self.power

def main():
    filename = input()

    data = np.loadtxt(filename, delimiter=',')
    box = np.reshape(data, (200,200,200))

    bins =np.logspace(np.log10(0.021), np.log10(3), num=13)

    k,delk = PowerSpectrum(box, bins = bins).compute_pspec()
    
    np.savetxt( "pspec_" + filename + '.csv', np.vstack((k,delk)), delimiter=',')

if __name__ == "__main__":
    main()