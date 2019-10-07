# Power Spectra

Functions that do power spectra related things. 

* **powerSpectrum.py** - returns the power spectrum of a given grid of data (flat sky). Input is in real space, either 3D grid or 2D. To run (unix), do: $ file_location | python powerSpectrum.py

Default bin edges are 13 logarithmically scaled bins from kspace values of 0.021 Mpc^-1 to 3 Mpc^-1, and returns delta squared quantity. If want regular power, give argument del_squared=False to compute_pspec method.
 
* **fftReturnReal.py** - returns a grid whose 2dfft returns real values only. Helper function for creating a field from a given power spectrum.

* **psToField.py** - given a function that describes a power spectrum, returns the corresponding field. 
