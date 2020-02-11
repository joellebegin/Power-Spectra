
import numpy as np
from numpy.fft import fftn, fftshift

class PowerSpectrum():

    def __init__(self, field, bins = None, L = 300, do_ft = True):
        '''
        Creates a power spectrum object, with methods that will output either a 
        regular 1d power spectrum or a cylindrical power spectrum. 

        Parameters
        ----------
        field: numpy ndarray
            field for which to compute the power spectrum

        bins: NoneType or 1darray
            if not None, fourier space bid edges for isotropic averaging. 
            the defauult is 13 logarithmically spaced bins as specified below

        L: int or float
            real space resolution of box in Mpc

        do_ft: Bool
            set to False if box is already in fourier space

        Methods
        -------
        compute_pspec: 
            returns the average k value going into each bin, and the power of 
            that bin

        cylindrical_pspec:
            to be implemented
        '''
        self.field = field
        
        #----------------------------- box specs ------------------------------#
        self.L = L
        self.ndims = len(field.shape)
        self.n = field.shape[0] #number of pixels along one axis
        self.survey_size = (self.n**self.ndims)#volume of box


        self.delta_k = 2*np.pi/self.L #kspace resolution of 1 pixel
        self.delta_r = self.L/self.n #real space resolution of 1 pixel

        #--------------------- Power spectrum attributes ----------------------#

        if bins is not None:
            self.bins = bins
        else: #default bins
            self.bins = np.logspace(np.log10(0.021), np.log10(3), num=13)

        if do_ft:
            self.fourier()
        else:
            self.field_fourier = self.field

        self.abs_squared = np.abs(self.field_fourier)**2 


    def fourier(self):
        fourier_transform = fftshift(fftn(fftshift(self.field)))
        scaled = fourier_transform*(self.delta_r**self.ndims)
        self.field_fourier = scaled

    #======================= METHODS THAT OUTPUT BOXES ========================#

    def compute_pspec(self, del_squared = True, ignore_0 = False):
        ''' computes the power spectrum. 

        Parameters
        ----------
        del_squared: Bool
            set to True if want the delta squared quantity rather than raw power

        ignore_0: Bool
            set to True if want to exclude the zero k vector in the averaging

        Returns
        -------
        average_k: 1darray
            the average value of k for the bins

        power: 1darray
            the power (or delta squared) of each bin
        '''
        self.ignore_0 = ignore_0
        self.del_squared = del_squared

        self.p_spec()
        return self.average_k, self.power


    #cylindrical pspec main will live here


    #=============== METHODS RELATED TO VANILLA POWER SPECTRUM ================#

    def p_spec(self):
            '''Main method of power spectrum compuation. Organizes functions.'''

            self.grid() #sets up grid of radial distances
            
            self.sort() #sorting radii + field vals (increasing)
            self.get_bin_ind() #indices determing which elements go into bins
            self.average_bins() #computing average of bins
            self.power = self.field_bins/self.survey_size
            
            if self.del_squared:
                self.power /= 2*np.pi**2

    

    def r3_norm(self,rx,ry,rz): #can make this obsolete with np.norm giving axis =0
        '''calculating length of each voxel's radial distance from origin'''
        return np.sqrt(rx**2 + ry**2 + rz**2)
    
    def grid(self):
        '''
        Generates a fourier space grid with spacing set by box specs, and finds 
        radial distance of each pixel from origin. Useful variables created:

        radii: numpy ndarray 
            grid that contains radial distance of each pixel from origin, 
            in kspace units
        '''
        
        origin = self.n//2
        self.rmax = self.n - 0.5 - origin

        #dont actually need to split these up, np.indices can be assigned to one var
        if self.ndims == 2: 
            x,y = np.indices(self.field.shape)
            self.radii = np.hypot(x - origin, y - origin)*self.delta_k
            
        elif self.ndims == 3: 
            x,y,z = np.indices(self.field.shape)
            self.radii = self.r3_norm(x-origin, y-origin, z-origin)*self.delta_k
        
        if self.del_squared:
                self.abs_squared *= self.radii**3
                
    def sort(self):
        ''' sorts radii, and the field value corresponding to each radius in 
        ascending order. sort_ind is here so as to not lose track of which 
        radius corresponds to which field value
        
        data is flattened in this step'''

        sort_ind = np.argsort(self.radii.flat)
        self.r_sorted = self.radii.flat[sort_ind]
        self.vals_sorted = self.abs_squared.flat[sort_ind] 

        if self.ignore_0: #excluding zero vector
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
    
