import numpy as np
from numpy.fft import fftn, fftshift
import numpy.linalg as LA

class PowerSpectrum():

    def __init__(self, field, bins = None, L = 300):
        '''
        - field: field to compute power spectrum (either real space or fourier space)
        - do_ft: if field is given in real space, set to True
        - bins: array, if want to give bin edges in fourier space units
        - bin_w: else, specify bin width in pixel units
        - L: real space length of box
        '''

        self.L = L
        
        self.ndims = len(field.shape)
        self.n = field.shape[0] #number of pixels along one axis

        self.delta_r = self.L/self.n #real space resolution of 1 pixel

        self.field = field
        self.field_fourier = self.fourier(field)

        if do_ft:
            self.field_fourier = self.fourier(field)
        else:
            self.field_fourier = field
        
        self.abs_squared = np.abs(self.field_fourier)**2

        
        self.abs_squared = np.abs(self.field_fourier)**2 #amplitude of field squared
        self.delta_k = 2*np.pi/self.L #kspace resolution of 1 pixel

        self.survey_size = (self.L**self.ndims)#volume of box
    
    def fourier(self,f):
        fourier_transform =  fftshift(fftn(fftshift(f)))
        scaled = fourier_transform*(self.delta_r**self.ndims)
        return scaled

    def r3_norm(self,rx,ry,rz):
        '''calculating length of each voxel's radial distance from origin'''
        return np.sqrt(rx**2 + ry**2 + rz**2)
    
        self.sort() #sorting radii + field vals (increasing)
        self.get_bin_ind() #indices determing which elements go into bins
        self.average_bins() #computing average of bins
        self.power = self.field_bins/self.survey_size #normalizing
        
        if del_squared:
            self.power *= (1/(2*np.pi**2))


    def get_bins(self):
        '''according to specifications, creates the bins with which the power 
        spectrum will be computed. 
        
        The default are 13 logarithmically scaled bins ranging from the min
        k vector to the max k vector as specified by grid spacing.'''

        self.rmax = self.n - self.origin        
        
        #uniformly spaced bins of given width
        if self.bin_w is not None:     
            self.bins = np.arange(self.bin_w,self.rmax, self.bin_w)*self.delta_k
            
        #default when nothing is given
        elif self.bins is None:
            self.bins = np.logspace(np.log10(0.021), np.log10(self.rmax*self.delta_k), num=13)


    def grid(self, del_squared):
        '''sets up kspace grid, where each pixel is assigned its distance from 
        the origin in kspace'''
        
        if self.ndims == 2: 
            x,y = np.indices(self.field_fourier.shape)
            self.radii = np.hypot(x - self.origin, y - self.origin)*self.delta_k
            
        elif self.ndims == 3: 
            x,y,z = np.indices(self.field_fourier.shape)
            self.radii = LA.norm((x-self.origin, y-self.origin, z-self.origin), 
                    axis = 0)*self.delta_k
        
        #if computing delta squared quantity, each pixel is multiplied by the 
        #cube of its distance from the origin
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
        
    def get_bin_ind(self, r = None):
        '''given the desired bin edges, determines the index of the last pixel
        going into this bin
        
        -r: if want to give another array to find bin indices for'''
        self.get_bins()
        
        if r is not None: 
            self.r_to_bin = r
        else:
            self.r_to_bin = self.r_sorted
        bin_ind = [0]
        
        for bin_val in self.bins:
            val = np.argmax( self.r_to_bin > bin_val)
            if val == 0: #ie bin_val > r_max
                val = len(self.r_to_bin)
            bin_ind.append(val-1)
        self.bin_ind = np.array(bin_ind)

    def average_bins(self):
        ''' puts things in bins, averages the bins. Returns:
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
    
    def cylindrical_pspec(self, del_squared = False):
        plane_pspec = []
        self.k_parallel = np.arange(-self.n//2, self.n//2, 1)*self.delta_k
        self.k_parallel_norm = np.abs(self.k_parallel)
        
        
        for plane in self.field_fourier:
            k,power = PowerSpectrum(plane, do_ft= False).compute_pspec(del_squared)
            plane_pspec.append(power)
            
        self.plane_pspec = np.array(plane_pspec)
        
        
        self.sort_ind = np.argsort(self.k_parallel_norm)
        self.k_parallel_sorted = self.k_parallel_norm[self.sort_ind]
        self.k_perp_sorted = self.plane_pspec[self.sort_ind]
        
        
        self.get_bin_ind(r = self.k_parallel_sorted)
        
        r_binned = []
        spectra_binned = []
        bin_dims = []
        for i in range(1, len(self.bin_ind)): #THIS IS SLOW AND UGLY. FIX ONE DAY
            r_binned.append(np.sum(self.k_parallel_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1]))
            spectra_binned.append(np.sum(self.k_perp_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1], axis = 0))
            bin_dims.append(len(self.k_parallel_sorted[self.bin_ind[i-1]:self.bin_ind[i]+1]))
        
        self.bin_dims = np.array(bin_dims)
        self.r_binned = np.array(r_binned)/self.bin_dims
        self.spectra_binned = np.array(spectra_binned)/self.bin_dims

def main():
    filename = input()

    data = np.loadtxt(filename, delimiter=',')
    box = np.reshape(data, (200,200,200))

    k,delk = PowerSpectrum(box).compute_pspec()
    
    np.savetxt( "pspec_" + filename + '.csv', np.vstack((k,delk)), delimiter=',')

if __name__ == "__main__":
    main()