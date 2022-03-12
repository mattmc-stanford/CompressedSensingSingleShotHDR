===========================================================================

This project contains the functions necessary for simulation of
single-shot SVE images, subsequent separation into N undersampled LDR
images of uniform exposure, inpainting of said LDR images via
compressed sensing techniques, and fusion of results into an HDR image.

These implement the Stanford University EE 367 final project report:
Single Shot HDR Imaging via Compressed Sensing, by Matthew A. McCready.

===========================================================================

The code is separated into 5 files with various functions contained therin.

simulateSVE.py: contains functions for simulation of a single-shot SVE
input image.

compressed_sensing_solver.py: contains various methods for inpainting the 
undersampled LDR images via compressed sensing.

compressed_sensing_MEF.py: handles organization and set up of compressed
sensing problem and its results.

hdr_methods.py: creates the HDR objects from the constructed MEF LDR set.

main.py: contains function for analyzing convergence, and the main script
for executing the image processing pipeline.

===========================================================================

The necessary datasets are contained within 3 folders.

hdr_data: contains the Stanford Memorial Church MEF dataset developed by
Debevec et al.
Paul E. Debevec and Jitendra Malik. Recovering High Dynamic Range
Radiance Maps from Photographs. In SIGGRAPH 97, August 1997.

Rovinia, and Flowers: contain two of the N = 3 exposure data sets developed
by Merianos et al.
Merianos I., Mitianoudis N., "A Hybrid Multiple Exposure Image Fusion 
Approach for HDR Image Synthesis", IEEE International Conference on Imaging
Systems and Techniques (IST 2016), Chania, Greece, October 2016.

===========================================================================

Additional files include a pretrained denoising convolutional neural network
(DnCNN) model and framework following the methods of Zhang et al. as well as
a set of functions for performing finite difference operations.
Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a Gaussian Denoiser: Residual
Learning of Deep CNN for Image Denoising. IEEE Transactions in Image 
Processing 2017;26:3142–3155

===========================================================================

Matthew A. McCready
mattmc@stanford.edu
https://profiles.stanford.edu/matt-mccready