# forward-virieux

Parallel numerical FD simulation for P-SV wave, using the velocity-stress formulation on a staggered grid from Virieux 1986.
Requires OpenMP, Armadillo and optionally LAPACK and OpenBLAS (to speed up Armadillo).


# Example results

This is the response of a homgeneous medium with a `blob' of higher mu (Lamé's second parameter) to a Ricker wavelet 
with a central frequency of 50 Hz. The domain is about half a kilometre transversally with 400 innner grid points. 
The boundary conditions are free surface (top) and no-reflecting boundaries on all other sides (Cerjan's paper).
![Ricker wavelet response](https://raw.githubusercontent.com/larsgeb/forward-virieux/master/examples/fileFWI.gif)
