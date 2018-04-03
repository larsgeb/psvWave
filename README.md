# forward-virieux

Parallel numerical FD simulation for P-SV wave, using the velocity-stress formulation on a staggered grid from Virieux 1986.
Requires OpenMP, Armadillo and optionally LAPACK and OpenBLAS (to speed up Armadillo).

The package is also able to compute kernels to be used in Full Waveform inversion. 

Most functions are self explanatory (I hope).

#### Examples

##### Kernels
Three material parameter kernels for a single source (Ricker Wavelet) with three receivers. 
The background model is perturbed by a Gaussian density blob exactly midway.
From top to bottom the kernels are density, Lamé's first and second parameter.

![Kernels](examples/kernels.png?raw=true)

##### Error in kernels
If one calculates the kernels and actually moves in that direction we can evaluate its error. 
By comparing the resulting misfit difference with the predicted change from a finite difference 
approximation we have an indication of the quality of the kernel.

What's more, we can get an indication of the minimum distance we need to traverse to get an 
actual misfit change. Below this distance (very small epsilon), machine precision prohibits 
any informative evaluation. The perturbation used in the figure below is exactly the negative 
of the original perturbation to compute the above kernels. Epsilon = 1 corresponds to retrieving 
the exact model.

![FD error](examples/fdTest.png?raw=true)