# Matthew A. McCready, Department of Electrical Engineering, Stanford University, 2022
# This python file contains functions for solving the compressed sensing problem Cx = b for inpainting undersampled
# images. It is a part of the SingleShotHDR project (Single-Shot HDR Imaging via Compressed Sensing) submitted in
# fulfillment of the EE367 Computational Imaging Final Project at Stanford University.

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from tqdm import tqdm
from time import time
from scipy.sparse.linalg import cg, LinearOperator
import torch
# check if GPU is available, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DnCNN model from Kai Zhag
from network_dncnn import DnCNN as net

# this function implements the finite differences method
from finite_differences import *

def admm_l1(b, C, lam, rho, num_iters, x0=[], target=[]):
    # ==================================================================================================================
    # This function is used to solve the compressed sensing problem Cx = b for x via basis pursuit denoising (BPDN) in
    # the Fourier domain with alternating direction method of multipliers (ADMM).
    # Inputs:
    #   b         - The input measurements. Note that b is not a column vector, but an array of the shape of the desired
    #               solution. [ndarray of solution shape]
    #   C         - The sampling/measurement matrix. Note that this is not an Mpixels x Mpixels matrix, but rather the
    #               shape of the desired solution. It will be used in ELEMENTWISE multiplication (less memory usage). C
    #               should be 1's at the measured pixels and 0's everywhere else. [ndarray of solution shape]
    #   lam       - Regularizer weighting lambda. [float]
    #   rho       - ADMM weighting parameter rho. [float]
    #   num_iters - Number of ADMM iterations to carry out. [int]
    #   x0        - Initial guess at a solution IN PRIMAL DOMAIN (default zeros). If an empty list is supplied x0
    #               defaults to all zeros. [ndarray of solution shape or empty list]
    #   target    - Ground truth for comparison (defaults to empty). If supplied, function will calculate convergence
    #               data. [ndarray of solution shape or empty list]
    #
    # Outputs:
    #   x         - The inpainted LDR image IN PRIMAL DOMAIN. [ndarray]
    #   timing    - Cumulative convergence timing at each iteration. Returns empty list if no target is supplied.
    #               [ndarray or empty list]
    #   residuals - The standardized error (summed absolute difference of solution with ground truth per pixel) for
    #               each iteration. Returns empty list if no target is supplied. [ndarray or empty list]
    # ==================================================================================================================

    # note that the elementwise implementation of C results that C transpose = C. Loss function is unaffected.
    Afun = lambda x: C * np.fft.ifft2(x, axes=(0, 1))  # for LinearOperator representation of b = Ax
    Atfun = lambda x: np.fft.fft2(C * x, axes=(0, 1))  # for LinearOperator representation of Afun conjugate transpose

    imageResolution = np.shape(b)  # get shape of solution

    if len(x0) == 0:  # defaults x0 to zeros
        x0 = np.zeros_like(b)

    # initialize x,u with all zeros (complex), initialize z with 2DFFT of x0
    x = np.fft.fft2(np.zeros_like(b), axes=(0, 1))
    z = np.fft.fft2(x0, axes=(0, 1))
    u = np.fft.fft2(np.zeros_like(b), axes=(0, 1))

    nels = np.size(b)  # number of elements in image

    efficiencyCheck = False  # default no interest in convergence
    timing = []
    residuals = []
    if np.array(target).any():  # we are interested in convergence, set flag to calculate
        efficiencyCheck = True
        residuals = np.zeros((num_iters, 1))
        timing = np.zeros((num_iters, 1))

    for it in tqdm(range(num_iters)):

        t0 = time()  # time at start of iter

        cg_iters = 25  # number of iterations for CG solver
        cg_tolerance = 1e-12  # convergence tolerance of cg solver

        # for x update
        v = z - u
        btilde = Atfun(b) + rho * v

        def mv(g):  # defines LinearOperator for x update
            g = np.reshape(g, imageResolution)
            out = Atfun(Afun(g)) + rho * g
            return out.flatten()
        Atilde = LinearOperator((nels, nels), matvec=mv)

        # update x using conjugate gradient solver
        x, _ = cg(Atilde, btilde.flatten(), tol=cg_tolerance, maxiter=cg_iters)
        x = np.reshape(x, imageResolution)

        # update z using elementwise soft thresholding
        kappa = lam / rho
        v = x + u
        z = np.maximum(1 - kappa / np.abs(v), 0) * v

        # update u
        u = u + x - z

        if efficiencyCheck:  # calculates time and standardized error for iteration if desired
            timing[it] = time() - t0
            est = np.abs(np.fft.ifft2(x, axes=(0, 1)))
            residuals[it] = np.sum(np.abs(target - est))/np.size(x)

    return np.fft.ifft2(x, axes=(0, 1)), np.cumsum(timing), residuals


def admm_DnCNN(b, C, lam, rho, num_iters, x0=[], target=[]):
    # ==================================================================================================================
    # This function is used to solve the compressed sensing problem Cx = b for x via a denoising convolutional neural
    # network (DnCNN) regularizer with alternating direction method of multipliers (ADMM).
    # Inputs:
    #   b         - The input measurements. Note that b is not a column vector, but an array of the shape of the desired
    #               solution. [ndarray of solution shape]
    #   C         - The sampling/measurement matrix. Note that this is not an Mpixels x Mpixels matrix, but rather the
    #               shape of the desired solution. It will be used in ELEMENTWISE multiplication (less memory usage). C
    #               should be 1's at the measured pixels and 0's everywhere else. [ndarray of solution shape]
    #   lam       - Regularizer weighting lambda. [float]
    #   rho       - ADMM weighting parameter rho. [float]
    #   num_iters - Number of ADMM iterations to carry out. [int]
    #   x0        - Initial guess at a solution (default zeros). If an empty list is supplied x0 defaults to all zeros.
    #               [ndarray of solution shape or empty list]
    #   target    - Ground truth for comparison (defaults to empty). If supplied, function will calculate convergence
    #               data. [ndarray of solution shape or empty list]
    #
    # Outputs:
    #   x         - The inpainted LDR image. [ndarray]
    #   timing    - Cumulative convergence timing at each iteration. Returns empty list if no target is supplied.
    #               [ndarray or empty list]
    #   residuals - The standardized error (summed absolute difference of solution with ground truth per pixel) for
    #               each iteration. Returns empty list if no target is supplied. [ndarray or empty list]
    # ==================================================================================================================

    # loads pre-trained DnCNN model
    model = net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
    model.load_state_dict(torch.load('dncnn_25.pth'), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # note that the elementwise implementation of C results that C transpose = C. Loss function is unaffected.
    Afun = lambda x: C * x  # for LinearOperator representation of b = Ax
    Atfun = lambda x: C * x  # for LinearOperator representation of Afun conjugate transpose

    imageResolution = np.shape(b)  # get shape of solution

    if len(x0) == 0:  # defaults x0 to zeros
        x0 = np.zeros_like(b)

    # initialize x,u with all zeros, initialize z as x0
    x = np.zeros_like(x0)
    z = x0.copy()
    u = np.zeros_like(x0)

    nels = np.size(b)  # number of elements in image

    efficiencyCheck = False  # default no interest in convergence
    timing = []
    residuals = []
    if np.array(target).any():  # we are interested in convergence, set flag to calculate
        efficiencyCheck = True
        residuals = np.zeros((num_iters, 1))
        timing = np.zeros((num_iters, 1))

    for it in tqdm(range(num_iters)):

        t0 = time()  # time at start of iter

        cg_iters = 25  # number of iterations for CG solver
        cg_tolerance = 1e-12  # convergence tolerance of cg solver

        # for x update
        v = z - u
        btilde = Atfun(b) + rho * v

        def mv(g):  # defines LinearOperator for x update
            g = np.reshape(g, imageResolution)
            out = Atfun(Afun(g)) + rho * g
            return out.flatten()
        Atilde = LinearOperator((nels, nels), matvec=mv)

        # update x using conjugate gradient solver
        x, _ = cg(Atilde, btilde.flatten(), tol=cg_tolerance, maxiter=cg_iters)
        x = np.reshape(x, imageResolution)

        # update z using DnCNN denoiser
        v = x + u
        z = np.zeros_like(v)
        for k in range(3):  # DnCNN takes each channel individually
            v_tensor = torch.reshape(torch.from_numpy(v[:,:,k]).float().to(device), (1, 1, v.shape[0], v.shape[1]))
            v_tensor_denoised = model(v_tensor)
            z[:, :, k] = torch.squeeze(v_tensor_denoised).cpu().numpy()

        # update u
        u = u + x - z

        if efficiencyCheck:  # calculates time and standardized error for iteration if desired
            timing[it] = time() - t0
            residuals[it] = np.sum(np.abs(target - x))/np.size(x)

    return x, np.cumsum(timing), residuals


def admm_tv(b, C, lam, rho, num_iters, x0=[], target=[], aniso_tv=True):
    # ==================================================================================================================
    # This function is used to solve the compressed sensing problem Cx = b for x via an anisotropic total variation (TV)
    # regularizer with alternating direction method of multipliers (ADMM). NOTE: This solver operates on a SINGLE COLOUR
    # CHANNEL at a time to simplify the implementation. Solutions etc. are of shape n x m, not n x m x 3.
    # Inputs:
    #   b         - The input measurements. Note that b is not a column vector, but an array of the shape of the desired
    #               solution. [ndarray of solution shape]
    #   C         - The sampling/measurement matrix. Note that this is not an Mpixels x Mpixels matrix, but rather the
    #               shape of the desired solution. It will be used in ELEMENTWISE multiplication (less memory usage). C
    #               should be 1's at the measured pixels and 0's everywhere else. [ndarray of solution shape]
    #   lam       - Regularizer weighting lambda. [float]
    #   rho       - ADMM weighting parameter rho. [float]
    #   num_iters - Number of ADMM iterations to carry out. [int]
    #   x0        - Initial guess at a solution (default zeros). If an empty list is supplied x0 defaults to all zeros.
    #               [ndarray of solution shape or empty list]
    #   target    - Ground truth for comparison (defaults to empty). If supplied, function will calculate convergence
    #               data. [ndarray of solution shape or empty list]
    #   aniso_tv  - Flag to use anisotropic TV (detault True). [boolean]
    #
    # Outputs:
    #   x         - The inpainted LDR image. [ndarray]
    #   timing    - Cumulative convergence timing at each iteration. Returns empty list if no target is supplied.
    #               [ndarray or empty list]
    #   residuals - The standardized error (summed absolute difference of solution with ground truth per pixel) for
    #               each iteration. Returns empty list if no target is supplied. [ndarray or empty list]
    # ==================================================================================================================

    # note that the elementwise implementation of C results that C transpose = C. Loss function is unaffected.
    Afun = lambda x: C * x  # for LinearOperator representation of b = Ax
    Atfun = lambda x: C * x  # for LinearOperator representation of Afun conjugate transpose

    if len(x0) == 0:
        x0 = np.zeros_like(b)

    imageResolution = np.shape(b)  # get shape of solution

    if len(x0) == 0:  # defaults x0 to zeros
        x0 = np.zeros_like(b)

    # initialize x,u with all zeros, initialize z as x0
    x = x0.copy()
    z = np.zeros((2, imageResolution[0], imageResolution[1]))
    z[0, :, :] = x0
    z[1, :, :] = x0
    u = np.zeros((2, imageResolution[0], imageResolution[1]))

    nels = np.size(b)  # number of elements in image

    efficiencyCheck = False  # default no interest in convergence
    timing = []
    residuals = []
    if np.array(target).any():  # we are interested in convergence, set flag to calculate
        efficiencyCheck = True
        residuals = np.zeros((num_iters, 1))
        timing = np.zeros((num_iters, 1))
        count = 0

    for it in tqdm(range(num_iters)):

        t0 = time()  # time at start of iter

        cg_iters = 25  # number of iterations for CG solver
        cg_tolerance = 1e-12  # convergence tolerance of cg solver

        # for x update
        v = z - u
        btilde = Atfun(b) + rho * opDtx(v)

        def mv(g):  # defines LinearOperator for x update
            g = np.reshape(g, imageResolution)
            out = Atfun(Afun(g)) + rho * opDtx(opDx(g))
            return out.flatten()
        Atilde = LinearOperator((nels, nels), matvec=mv)

        # update x using conjugate gradient solver
        x, _ = cg(Atilde, btilde.flatten(), tol=cg_tolerance, maxiter=cg_iters)
        x = np.reshape(x, imageResolution)

        # update z using soft thresholding
        kappa = lam / rho
        v = opDx(x) + u
        if aniso_tv:  # proximal operator of anisotropic TV term
            z = np.maximum(1 - kappa / np.abs(v), 0) * v
        else:  # proximal operator of isotropic TV term
            vnorm = np.sqrt(v[0, :, :] ** 2 + v[1, :, :] ** 2)
            z[0, :, :] = np.maximum(1 - kappa / vnorm, 0) * v[0, :, :]
            z[1, :, :] = np.maximum(1 - kappa / vnorm, 0) * v[1, :, :]

        # update u
        u = u + opDx(x) - z

        if efficiencyCheck:  # calculates time and standardized error for iteration if desired
            timing[it] = time() - t0
            residuals[it] = np.sum(np.abs(target - x))/np.size(x)
            count+=1

    return x, np.cumsum(timing), residuals


def default_solver(b, C, lam, num_iters, learning_rate=5e-2, x0=[], target=[]):
    # ==================================================================================================================
    # This function is used to solve the compressed sensing problem Cx = b for x via basis pursuit denoising (BPDN) in
    # the Fourier domain with the PyTorch Adam solver.
    # Inputs:
    #   b         - The input measurements. Note that b is not a column vector, but an array of the shape of the desired
    #               solution. [ndarray of solution shape]
    #   C         - The sampling/measurement matrix. Note that this is not an Mpixels x Mpixels matrix, but rather the
    #               shape of the desired solution. It will be used in ELEMENTWISE multiplication (less memory usage). C
    #               should be 1's at the measured pixels and 0's everywhere else. [ndarray of solution shape]
    #   lam       - Regularizer weighting lambda. [float]
    #   num_iters - Number of Adam iterations to carry out. [int]
    #   learning_rate - Learning rate of Adam solver (default 5e-2). [float]
    #   x0        - Initial guess at a solution (default zeros). If an empty list is supplied x0 defaults to all zeros.
    #               [ndarray of solution shape or empty list]
    #   target    - Ground truth for comparison (defaults to empty). If supplied, function will calculate convergence
    #               data. [ndarray of solution shape or empty list]
    #
    # Outputs:
    #   x         - The inpainted LDR image. [ndarray]
    #   timing    - Cumulative convergence timing at each iteration. Returns empty list if no target is supplied.
    #               [ndarray or empty list]
    #   residuals - The standardized error (summed absolute difference of solution with ground truth per pixel) for
    #               each iteration. Returns empty list if no target is supplied. [ndarray or empty list]
    # ==================================================================================================================

    # convert inputs to PyTorch tensors
    b = torch.from_numpy(b).to(device)
    C = torch.from_numpy(C).to(device)

    # initialize x and convert to PyTorch tensor
    if len(x0) == 0:
        x0 = np.zeros_like(b)
    x = torch.fft.fft2(torch.from_numpy(x0), dim=(0, 1)).to(device)
    x.requires_grad = True

    # initialize Adam optimizer
    optim = torch.optim.Adam(params=[x], lr=learning_rate)

    Afun = lambda x: C * torch.fft.ifft2(x, dim=(0, 1))  # define operator A in Ax = b
    reg_fn = lambda x: torch.abs(x).sum()  # define regularizer term

    efficiencyCheck = False  # default no interest in convergence
    timing = []
    residuals = []
    if np.array(target).any():  # we are interested in convergence, set flag to calculate
        efficiencyCheck = True
        target = torch.from_numpy(target).to(device)
        residuals = torch.zeros((num_iters, 1))
        timing = np.zeros((num_iters, 1))
        count = 0

    for it in tqdm(range(num_iters)):

        t0 = time()  # time at start of iter

        # set all gradients of the computational graph to 0
        optim.zero_grad()

        # this term computes the data fidelity term of the loss function
        loss_data = (torch.abs(Afun(x) - b)).pow(2).sum()

        # this term computes the reluzarizer term of the loss function
        loss_regularizer = reg_fn(x)

        # compute weighted sum of data fidelity and regularization term
        loss = loss_data + lam * loss_regularizer

        # compute backwards pass
        loss.backward()

        # take a step with the Adam optimizer
        optim.step()

        if efficiencyCheck:  # calculates time and standardized error for iteration if desired
            timing[it] = time() - t0
            residuals[it] = torch.sum(torch.abs(target - torch.abs(torch.fft.ifft2(x, dim=(0, 1)))))/torch.numel(x)
            count+=1

    x = torch.fft.ifft2(x, dim=(0, 1))  # convert solution back to primal domain

    if efficiencyCheck:  # return convergence data to ndarray if desired
        residuals = residuals.detach().cpu().numpy()

    return x.detach().cpu().numpy(), np.cumsum(timing), residuals


