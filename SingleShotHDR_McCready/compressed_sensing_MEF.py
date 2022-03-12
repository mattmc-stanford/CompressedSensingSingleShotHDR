# Matthew A. McCready, Department of Electrical Engineering, Stanford University, 2022
# This python file contains functions for setting up N compressed sensing problems Cx = b for inpainting N undersampled
# images of compression factor N. It is a part of the SingleShotHDR project (Single-Shot HDR Imaging via Compressed
# Sensing) submitted in fulfillment of the EE367 Computational Imaging Final Project at Stanford University.

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

# DnCNN model from Kai Zhag
from network_dncnn import DnCNN as net

from compressed_sensing_solver import default_solver, admm_l1, admm_DnCNN, admm_tv
from simulateSVE import generate_mask, simulate_SVE_image

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def full_method(sve, mask, ldr_set, num_iters=500, lam=1.0, rho=1.0, solver='admml1', denoise=False, efficiency=False, x0=[]):
    # ==================================================================================================================
    # This function is used to set up and receive solutions to the N compressed sensing problems for inpainting LDR
    # images. It does not directly solve the CS problem but serves as an intermediate step to the solvers themselves.
    # Inputs:
    #   sve        - The spatially varying exposure input image of N exposures. [ndarray]
    #   mask       - The grayscale mask that denotes the exposures for the sve image. The values of mask are NOT the
    #                exposures but whole-numbers 1, ... ,N that correspond to exposures in the main function. [ndarray]
    #   ldr_set    - The list of true ldr images at exposures corresponding to the values of mask. This is only used
    #                for determining the number of exposures and for convergence tests if desired. [list of ndarrays]
    #   num_iters  - The number of iterations to perform while solving (default 500) [int]
    #   lam        - The lambda hyper parameter value (default 1.0) [float]
    #   rho        - The rho hyper parameter value (default 1.0) [float]
    #   solver     - The solver to be used (default 'admml1'). Options are 'admml1', 'admmDnCNN', 'admmTV', 'adam'. This
    #                input is NOT case-sesitive. [string]
    #   denoise    - Logical flag for whether or not to apply the DnCNN to the inpainted LDR images (default False).
    #                This can improve results if desired. Not presented in report or presentation. [boolean]
    #   efficiency - Logical flag for whether or not to test and record convergence data (default False). [boolean]
    #   x0         - The initial guess at a solution for the solver (default zeros). If an empty list is supplied, x0
    #                defaults to zeros. Otherwise an ndarray can be supplied or a list of ndarrays with length N
    #                exposures. [empty list, ndarray, or list of ndarrays]
    #
    # Outputs:
    #   recon_set  - The inpainted ldr images as a list, ordered by exposure number. [list of ndarrays]
    #   timing     - The convergence timing data as a list corresponding to images of recon_set. Empty if efficiency was
    #                set to False. [list of ndarrays]
    #   residuals  - The standardized errors (summed absolute difference of solution with ground truth per pixel) for
    #                each inpainted image in recon_set. Empty if efficiency was set to False. [list of ndarrays]
    # ==================================================================================================================

    num_exposures = len(ldr_set)  # get number of exposures
    recon_set = [None] * num_exposures  # initialize outputs
    timing = [None] * num_exposures
    residuals = [None] * num_exposures
    target = []  # only used if efficiency set to True

    for k1 in range(num_exposures):  # loop over number of LDR images to inpaint

        if len(x0) > 0 and isinstance(x0, list):  # handles if multiple x0's are supplied
            x0_iter = x0[k1]
        else:
            x0_iter = x0

        C = np.zeros_like(sve)  # create measurement matrix for exposure k1
        C[mask == (k1 + 1)] = 1

        if efficiency:  # check if we care about residuals and timing
            target = ldr_set[k1]

        # solving problem k1 with the desired solver
        if solver.lower() == 'admml1':  # uses ADMM for BPDN
            ldr_rec, timing[k1], residuals[k1] = admm_l1(C * sve, C, lam, rho, num_iters, target=target, x0=x0_iter)
        elif solver.lower() == 'admmdncnn':  # uses ADMM with DnCNN regularizer
            ldr_rec, timing[k1], residuals[k1] = admm_DnCNN(C * sve, C, lam, rho, num_iters, target=target, x0=x0_iter)
        elif solver.lower() == 'admmtv':  # uses ADMM with TV regularizer (acts on one colour channel at a time)
            ldr_rec = np.zeros_like(sve)
            tim = 0  # must sum timings across each colour channel for fair comparison
            res = 0  # must average timings across each colour channel for fair comparison
            targett = []  # single colour channel of target
            x0t = []  # single colour channel of initial guess
            for k2 in range(3):  # loop over colour channels
                Ct = C[:, :, k2]  # each of these are actually the same. Just makes the necessary dimensions.
                svet = sve[:, :, k2]  # single colour channel of sve
                if efficiency:
                    targett = target[:, :, k2]
                if len(x0) > 0:
                    x0t = x0_iter[:, :, k2]
                ldr_rec[:, :, k2],timing[k1], residuals[k1] = admm_tv(Ct * svet, Ct, lam, rho, num_iters, target=targett, x0=x0t)
                if efficiency:
                    tim += timing[k1]
                    res += residuals[k1]/3
            timing[k1] = tim
            residuals[k1] = res
        elif solver.lower() == 'adam':  # uses default Adam solver for BPDN
            ldr_rec, timing[k1], residuals[k1] = default_solver(C * sve, C, lam, num_iters, learning_rate=5, target=target, x0=x0_iter)
        else:  # invalid input. Throws a crude error.
            print('NOT A SOLVER')
            KILL = [1, 2]
            KILL[100]  # throws an error

        ldr = np.abs(ldr_rec)  # in case BPDN returns complex values

        if denoise:  # can apply DnCNN to inpainted ldr image
            # load pre-trained DnCNN model
            model = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
            model.load_state_dict(torch.load('dncnn_25.pth'), strict=True)
            model.eval()
            for k, v in model.named_parameters():
                v.requires_grad = False
            model = model.to(device)

            for k in range(3):  # denoise each colour channel
                ldrc_tensor = torch.reshape(torch.from_numpy(ldr[:,:,k]).float().to(device), (1, 1, ldr.shape[0], ldr.shape[1]))
                ldrc_tensor_dn = model(ldrc_tensor)
                ldr[:, :, k] = torch.squeeze(ldrc_tensor_dn).cpu().numpy()

        recon_set[k1] = ldr  # record inpainted ldr

    return recon_set, timing, residuals