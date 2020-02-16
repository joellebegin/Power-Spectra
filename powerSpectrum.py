
import numpy as np
from numpy.fft import fftn, fftshift
from numpy.linalg import norm

class PowerSpectrum():

    def __init__(self, field, k_bins = None, n_bins = None, L = 300, do_ft = True):
        '''
        Creates a power spectrum object, with methods that will output either a 
        regular 1d power spectrum or a cylindrical power spectrum. 

        Parameters
        ----------
        field: numpy ndarray
            field for which to compute the power spectrum

        bins: NoneType or 1darray
            if not None, fourier space bin edges for isotropic averaging. 
            the defauult is 13 logarithmically spaced bins as specified below

        n_bins: NoneType or int
            if want uniformly spaced bins, this will generate bin edges from 
            the minimum k to max k as specified by box specs

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
        self.k_bins = k_bins 
        self.n_bins = n_bins
        
        #----------------------------- box specs ------------------------------#
        self.L = L
        self.ndims = len(field.shape)
        self.n = field.shape[0] #number of pixels along one axis
        self.survey_size = (self.n**self.ndims)#volume of box
        self.origin = self.n//2


        self.delta_k = 2*np.pi/self.L #kspace resolution of 1 pixel
        self.delta_r = self.L/self.n #real space resolution of 1 pixel

        self.rmax = (self.n - self.origin)*self.delta_k #max radius

        #--------------------- Power spectrum attributes ----------------------#

        self.get_bins() #defining bins variable according to specifications

        if do_ft: 
            self.fourier()
        else:
            self.field_fourier = self.field

        self.abs_squared = np.abs(self.field_fourier)**2 

    #============================== INIT METHODS ==============================#
    
    def fourier(self):
        fourier_transform = fftshift(fftn(fftshift(self.field)))
        scaled = fourier_transform*(self.delta_r**self.ndims)
        self.field_fourier = scaled

    def get_bins(self):
        
        if self.k_bins is not None:
            self.bins = self.k_bins

        elif self.n_bins is not None:
            self.bins = np.linspace(self.delta_k, self.rmax, self.n_bins)

        else: #default bins
            self.bins = np.logspace(np.log10(0.021), np.log10(self.rmax), num=13)

    #======================= METHODS THAT OUTPUT BOXES ========================#

    def compute_pspec(self, del_squared = True, ignore_0 = False, return_k = True):
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

        if return_k:
            return self.average_k, self.power
        else:
            return self.power

    def compute_cylindrical_pspec(self, ignore_0 = False, k_perp_bins = None,
    k_par_bins = None, delta_squared = False):

        self.k_perp_bins = k_perp_bins
        self.k_par_bins = k_par_bins
        self.delsq = delta_squared
        self.ignore_0 = ignore_0 

        self.cyl_pspec()


    #=============== METHODS RELATED TO VANILLA POWER SPECTRUM ================#

    def p_spec(self):
            '''Main method of power spectrum compuation. Organizes functions.'''

            self.grid() #sets up grid of radial distances
            
            self.sort() #sorting radii + field vals (increasing)
            
            #indices determing which elements go into bins
            self.get_bin_ind(self.bins, self.r_sorted)
            self.average_bins() #computing average of bins
            self.power = self.field_bins/self.survey_size
            
            if self.del_squared:
                self.power /= 2*np.pi**2


    def grid(self):
        '''
        Generates a fourier space grid with spacing set by box specs, and finds 
        radial distance of each pixel from origin. Useful variable created:

        radii: numpy ndarray 
            grid that contains radial distance of each pixel from origin, 
            in kspace units
        '''

        indices = (np.indices(self.field.shape) - self.origin)*self.delta_k
        self.radii = norm(indices, axis = 0)
        
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
        
    
#=============== METHODS RELATED TO CYLINDRICAL POWER SPECTRUM ================#

    def cyl_pspec(self):

        self.compute_kperp_pspec()
        self.bin_kpar()

    def compute_kperp_pspec(self):
        k_par_power = []
        for k_perp_slice in self.field:
            spec = PowerSpectrum(k_perp_slice, k_bins = self.k_par_bins, do_ft= False)
            power = spec.compute_pspec(del_squared= self.delsq, 
                                        ignore_0= self.ignore_0, return_k= False)
            k_par_power.append(power)
        
        self.k_par_power = np.array(k_par_power)

    def bin_kpar(self):
        self.k_par = np.arange(self.n//2,self.n//2)*self.delta_k
        #have to modify get_bin_ind so that it can take the bins and the radii 
        #as arugment, that way i can use it here

        #and average bins as well


#============================ BINNING FUNCTIONS ===============================#

    def get_bin_ind(self, bins, values):
            '''given the bin edges (bins), and data to be binned (values)
            determines value of last pixel going into each bin'''
            bin_ind = [0]
            for bin_val in bins:
                val = np.argmax( values > bin_val)
                if val == 0: #ie bin_val > r_max
                    val = len(values)
                else:
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