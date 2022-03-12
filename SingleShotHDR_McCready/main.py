# Matthew A. McCready, Department of Electrical Engineering, Stanford University, 2022
# This python file contains a main function for the overall pipeline of simulating a single-shot SVE image, producing N
# full LDR images from that SVE, and calculating an HDR image from that LDR set. It also contains a function for
# analyzing the convergence of various compressed sensing solvers, as well as a function for applying a zero-order hold
# to fill in an undersampled image. It is a part of the SingleShotHDR project (Single-Shot HDR Imaging via Compressed
# Sensing) submitted in fulfillment of the EE367 Computational Imaging Final Project at Stanford University.


import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import imageio
import cv2

from pdb import set_trace
from pathlib import Path

# my functions
from simulateSVE import generate_mask, simulate_SVE_image
from hdr_methods import multi_expsr_fusion
from compressed_sensing_MEF import full_method


def zero_hold(A):
    # ==================================================================================================================
    # This function applies a zero-order hold to a sampled image, filling in zero's with the values preceeding them.
    # Inputs:
    #   A - The image to be filled. Must be a single colour channel (i.e. shape m x n). [ndarray]
    #
    # Outputs:
    #   A  - The filled image. [ndarray]
    # ==================================================================================================================
    nrows = A.shape[0]
    ncols = A.shape[1]
    for r in range(nrows):  # for each element in A
        for c in range(ncols):
            if A[r, c] == 0:  # we need to fill this value
                idx = np.nonzero(A[1:r, c])  # get indices of non-zero elements preceding [r, c] in this row
                if idx[0].any():  # if there is a non-zero preceding value
                    ridx = idx[0][-1]  # fill with the last (closest) value
                    A[r, c] = A[ridx, c]
                else:  # there is no preceding non-zero value
                    idx = np.nonzero(A[r:, c])  # get the following non-zero values
                    ridx = idx[0][0]  # fill with the first (closest) value
                    A[r, c] = A[ridx+r, c]
    return A


def convergence(sve, mask, ldr):
    # ==================================================================================================================
    # This function is used to evaluate the convergence of the various solvers used in this work with different initial
    # guesses at a solution for inpainting LDR images.
    # Inputs: sve  - The spatially varying exposure input image of N exposures.
    #         mask - The grayscale mask that denotes the exposures for the sve image. The values of mask are NOT the
    #                expousres. They are whole-numbers 1, ... ,N that correspond to exposures in the main function.
    #         ldr  - A list containing the one ldr image to test inpainting of. Do NOT supply more than a single LDR
    #                image in this list.
    #
    # Outputs: None. This function produces several convergence plots, and when the raw data was desired the function
    #          was run in debugging mode and the arrays saved manually for later use.
    # ==================================================================================================================

    # generating initial guesses or starting points for solvers
    x0 = [None] * 4  # list of guesses
    x0[0] = np.zeros_like(sve)  # first is all zeros
    x0[1] = 0.5 * np.ones_like(sve)  # second is half-intensity (assuming ldr and sve are in range [0, 1])
    sigma = np.std(sve[mask == 1])  # third is sampled from Gaussian of std and mean measured values
    mu = np.mean(sve[mask == 1])
    x0[2] = np.clip(np.random.normal(mu, sigma, sve.shape), 0, 1)
    sve_1 = np.zeros_like(sve)  # fourth is a zero-order hold applied to the measured values
    sve_1[mask == 1] = sve[mask == 1]
    for k in range(3):
        sve_1[:, :, k] = zero_hold(sve_1[:, :, k])
    x0[3] = sve_1.copy()

    # initializing the sets of inpainted ldrs, timings for convergence, and standardized losses, for each solver
    recon_set1 = [None] * len(x0)
    timing1 = [None] * len(x0)
    residuals1 = [None] * len(x0)
    recon_set2 = [None] * len(x0)
    timing2 = [None] * len(x0)
    residuals2 = [None] * len(x0)
    recon_set3 = [None] * len(x0)
    timing3 = [None] * len(x0)
    residuals3 = [None] * len(x0)
    recon_set4 = [None] * len(x0)
    timing4 = [None] * len(x0)
    residuals4 = [None] * len(x0)

    # labels for plot legends
    labels = ['zeros', 'half intensity', 'gaussian', 'zero order hold']
    solvers = ['ADMM_L1', 'ADMM_DnCNN', 'ADMM_TV', 'Adam_L1']

    # iteration plot x-cutoff (Adam takes many interations)
    iters_cutoff = 300

    # enforce first loss to be calculated directly from x0
    init_loss = [None] * len(x0)
    for k in range(len(x0)):
        init_loss[k] = np.sum(np.abs(ldr[0] - x0[k])) / np.size(ldr[0])

    # ADMM BPDN solver==================================================================================================
    num_iters = 300
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for k in range(len(x0)):
        recon_set, timing, residuals = full_method(sve, mask, ldr, num_iters=num_iters, solver='admml1',
                                                               efficiency=True, x0=x0[k])
        residuals[0][0] = init_loss[k]
        timing[0] = np.insert(timing[0][:-1], 0, 0)
        recon_set1[k], timing1[k], residuals1[k] = (np.abs(recon_set[0]), timing[0], residuals[0])
        ax1[0].semilogy(np.arange(0, num_iters), residuals1[k], label=labels[k])
        ax1[1].semilogy(timing1[k], residuals1[k], label=labels[k])

    ax1[0].set_xlim(0, num_iters)
    ax1[1].set_xlim(0)
    ax1[0].legend()
    ax1[1].legend()
    ax1[0].set_xlabel('Iterations')
    ax1[0].set_ylabel('Standardized Loss')
    ax1[1].set_xlabel('Time (s)')
    ax1[1].set_ylabel('Standardized Loss')
    fig1.suptitle('Initialization Convergence for ADMM_L1')

    # ADMM DnCNN solver=================================================================================================
    num_iters = 50

    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    for k in range(len(x0)):
        recon_set, timing, residuals = full_method(sve, mask, ldr, num_iters=num_iters, solver='admmDnCNN',
                                                               rho=0.0001, efficiency=True, x0=x0[k])
        residuals[0][0] = init_loss[k]
        timing[0] = np.insert(timing[0][:-1], 0, 0)
        recon_set2[k], timing2[k], residuals2[k] = (recon_set[0], timing[0], residuals[0])
        ax2[0].semilogy(np.arange(0, num_iters), residuals2[k], label=labels[k])
        ax2[1].semilogy(timing2[k], residuals2[k], label=labels[k])

    ax2[0].set_xlim(0, num_iters)
    ax2[1].set_xlim(0)
    ax2[0].legend()
    ax2[1].legend()
    ax2[0].set_xlabel('Iterations')
    ax2[0].set_ylabel('Standardized Loss')
    ax2[1].set_xlabel('Time (s)')
    ax2[1].set_ylabel('Standardized Loss')
    fig2.suptitle('Initialization Convergence for ADMM_DnCNN')
    ''
    # ADMM tv solver====================================================================================================
    num_iters = 50
    fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for k in range(len(x0)):
        recon_set, timing, residuals = full_method(sve, mask, ldr, num_iters=num_iters, solver='admmtv', lam=0.005,
                                                   rho=1, efficiency=True, x0=x0[k])
        residuals[0][0] = init_loss[k]
        timing[0] = np.insert(timing[0][:-1], 0, 0)
        recon_set3[k], timing3[k], residuals3[k] = (recon_set[0], timing[0], residuals[0])
        ax3[0].semilogy(np.arange(0, num_iters), residuals3[k], label=labels[k])
        ax3[1].semilogy(timing3[k], residuals3[k], label=labels[k])

    ax3[0].set_xlim(0, num_iters)
    ax3[1].set_xlim(0)
    ax3[0].legend()
    ax3[1].legend()
    ax3[0].set_xlabel('Iterations')
    ax3[0].set_ylabel('Standardized Loss')
    ax3[1].set_xlabel('Time (s)')
    ax3[1].set_ylabel('Standardized Loss')
    fig3.suptitle('Initialization Convergence for ADMM_TV')

    # Adam BPDN solver==================================================================================================
    num_iters = 2500
    fig4, ax4 = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    for k in range(len(x0)):
        recon_set, timing, residuals = full_method(sve, mask, ldr, num_iters=num_iters, solver='adam', lam=0.0005,
                                                   efficiency=True, x0=x0[k])
        residuals[0][0] = init_loss[k]
        timing[0] = np.insert(timing[0][:-1], 0, 0)
        recon_set4[k], timing4[k], residuals4[k] = (np.abs(recon_set[0]), timing[0], residuals[0])
        ax4[0].semilogy(np.arange(0, num_iters), residuals4[k], label=labels[k])
        ax4[1].semilogy(timing4[k], residuals4[k], label=labels[k])

    ax4[0].set_xlim(0, num_iters)
    ax4[1].set_xlim(0)
    ax4[0].legend()
    ax4[1].legend()
    ax4[0].set_xlabel('Iterations')
    ax4[0].set_ylabel('Standardized Loss')
    ax4[1].set_xlabel('Time (s)')
    ax4[1].set_ylabel('Standardized Loss')
    fig4.suptitle('Initialization Convergence for Adam_L1')

    # plotting fastest solutions together
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].semilogy(np.arange(0, len(timing1[3])), residuals1[3], label=solvers[0])
    ax[0].semilogy(np.arange(0, len(timing2[3])), residuals2[3], label=solvers[1])
    ax[0].semilogy(np.arange(0, len(timing3[0])), residuals3[0], label=solvers[2])
    ax[0].semilogy(np.arange(0, len(timing4[3])), residuals4[3], label=solvers[3])
    ax[1].semilogy(timing1[3], residuals1[3], label=solvers[0])
    ax[1].semilogy(timing2[3], residuals2[3], label=solvers[1])
    ax[1].semilogy(timing3[0], residuals3[0], label=solvers[2])
    ax[1].semilogy(timing4[3], residuals4[3], label=solvers[3])

    ax[0].set_xlim(0, iters_cutoff)
    ax[1].set_xlim(0)

    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Standardized Loss')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Standardized Loss')
    fig.suptitle('Convergence for Various Solvers')

    # displaying resulting LDR inpaintings from fastest solutions
    fig6, ax6 = plt.subplots(nrows=2, ncols=2)
    ax6[0, 0].imshow(recon_set1[3])
    ax6[0, 1].imshow(recon_set2[3])
    ax6[1, 0].imshow(recon_set3[0])
    ax6[1, 1].imshow(recon_set4[3])

    ax6[0, 0].title.set_text('ADMM_L1')
    ax6[0, 1].title.set_text('ADMM_DnCNN')
    ax6[1, 0].title.set_text('ADMM_TV')
    ax6[1, 1].title.set_text('Adam_L1')

    ax6[0, 0].axis('off')
    ax6[0, 1].axis('off')
    ax6[1, 0].axis('off')
    ax6[1, 1].axis('off')

    return None


if __name__ == '__main__':
    # ==================================================================================================================
    # This script is used to call functions for simulating a single-shot SVE image, constructing N full LDR images from
    # said SVE, and fusing those to create an HDR image and tonemap it. It can also be used to check convergence of
    # compressed sensing solvers if lines are commented/uncommented.
    # ==================================================================================================================

    # loading in the LDR data to be used for single-shot HDR imaging (these will form the SVE).

    # Memorial Church multi-exposure data ==============================================================================
    hdr_dir = Path("hdr_data")
    # exposure times for memorial church HDR data (in order starting at file number 61)
    expsr = 1 / np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    file_nums = [61, 66, 71]  # N = 3 exposure set used in report
    # file_nums = [63, 66, 69, 72]  # N = 4 exposure set used in report

    expsr = expsr[np.array(file_nums).astype(int)-61]  # get relevant exposure times
    num_exposures = len(file_nums)  # number of exposures being used
    ldr_set = [None] * num_exposures  # initialize storage for ldr images

    for k, num in enumerate(file_nums):  # load in images
        filename = r"hdr_data\memorial00" + str(num) + ".png"
        ldr = io.imread(filename).astype(float) / 255  # scale to [0, 1]
        ldr = ldr[29:720, 19:480, :]  # crop out blue borders from alignment
        ldr_set[k] = ldr

    # ==================================================================================================================

    # Merianos et al. multi-exposure data set ==========================================================================
    # (Only used with Merten's et al exposure fusion as no exposure times were supplied with this dataset)
    '''
    # Rovinia dataset
    file_nums = [1, 2, 3]
    expsr = [1, 2, 3]  # random and unused. See above comment.
    num_exposures = len(file_nums)  # N = 3 exposures
    ldr_set = [None] * num_exposures  # store ldr images
    for k, num in enumerate(file_nums):  # load in images
        filename = r"Rovinia\rovinia" + str(num) + ".jpg"
        ldr = io.imread(filename).astype(float) / 255  # scale to [0, 1]
        ldr_set[k] = ldr
    '''
    '''
    # Flowers dataset
    file_nums = [1, 2, 3]
    expsr = [1, 2, 3]  # random and unused. See above comment.
    num_exposures = len(file_nums)  # N = 3 exposures
    ldr_set = [None] * num_exposures  # store ldr images
    for k, num in enumerate(file_nums):  # load in images
        filename = r"Flowers\Flowers" + str(num) + ".jpg"
        ldr = io.imread(filename).astype(float) / 255  # scale to [0, 1]
        ldr_set[k] = ldr
    '''
    # ==================================================================================================================

    # generate random mask and SVE for single-shot HDR
    mask = generate_mask(np.shape(ldr_set[0][:, :, 0]), num_exposures)
    sve = simulate_SVE_image(ldr_set, mask)

    # convergence(sve, mask, [ldr_set[0]])  # used if interested in convergence of solvers

    # creating potential initial guesses ===============================================================================

    # zero-order hold of each measurement
    zr_hold = [None] * num_exposures
    for k1 in range(num_exposures):
        sve_1 = np.zeros_like(sve)
        sve_1[mask == (k1+1)] = sve[mask == (k1+1)]
        for k2 in range(3):
            sve_1[:, :, k2] = zero_hold(sve_1[:, :, k2])
        zr_hold[k1] = sve_1.copy()

    # half-intensity
    x0 = 0.5 * np.zeros_like(sve)

    # Gaussian sampled initial guess for each exposure
    gauss = [None] * num_exposures
    for k1 in range(num_exposures):
        sigma = np.zeros_like(sve)
        sigma = np.std(sve[mask == 1])
        mu = np.mean(sve[mask == 1])
        gauss[k1] = np.clip(np.random.normal(mu, sigma, sve.shape), 0, 1)

    # producing and saving the inpaintings and hdr images===============================================================

    num = 0  # set to keep file names from a specific test together

    # Solving for hdr image with ADMM_L1
    recon_set1, timing, residuals = full_method(sve, mask, ldr_set, num_iters=400, solver='admml1', x0=zr_hold)
    filenames = [r'hdr_recons\admml1_scaledTonemap'+str(num), r'hdr_recons\admml1_tonemapped'+str(num), r'hdr_recons\admml1_hdr'+str(num)]
    hdr = multi_expsr_fusion(recon_set1, expsr, filenames)
    recon_set = np.save(r'ldr_inpainting_sets\admml1_recon_set'+str(num), np.array(recon_set1),)

    # Solving for hdr image with ADMM_DnCNN
    recon_set2, timing, residuals = full_method(sve, mask, ldr_set, num_iters=75, solver='admmDnCNN', rho=0.0001, x0=zr_hold)
    filenames = [r'hdr_recons\admmDnCNN_scaledTonemap'+str(num), r'hdr_recons\admmDnCNN_tonemapped'+str(num), r'hdr_recons\admmDnCNN_hdr'+str(num)]
    hdr = multi_expsr_fusion(recon_set2, expsr, filenames)
    recon_set = np.save(r'ldr_inpainting_sets\admmDnCNN_recon_set'+str(num), np.array(recon_set2))
    
    # Solving for hdr image with ADMM_TV
    recon_set3, timing, residuals = full_method(sve, mask, ldr_set, num_iters=20, solver='admmtv', lam=0.002, rho=1, x0=x0)
    filenames = [r'hdr_recons\admmTV_scaledTonemap'+str(num), r'hdr_recons\admmTV_tonemapped'+str(num),r'hdr_recons\admmTV_hdr'+str(num)]
    hdr = multi_expsr_fusion(recon_set3, expsr, filenames)
    recon_set = np.save(r'ldr_inpainting_sets\admmTV_recon_set'+str(num), np.array(recon_set3))

    # Solving for hdr image with ADAM_L1
    recon_set4, timing, residuals = full_method(sve, mask, ldr_set, num_iters=500, solver='adam',  lam=0.0005, x0=zr_hold)
    filenames = [r'hdr_recons\adaml1_scaledTonemap'+str(num), r'hdr_recons\adaml1_tonemapped'+str(num), r'hdr_recons\adaml1_hdr'+str(num)]
    hdr = multi_expsr_fusion(recon_set4, expsr, filenames)
    recon_set = np.save(r'ldr_inpainting_sets\adaml1_recon_set'+str(num), np.array(recon_set4))

