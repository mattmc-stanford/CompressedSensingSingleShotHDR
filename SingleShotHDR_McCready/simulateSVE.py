# Matthew A. McCready, Department of Electrical Engineering, Stanford University, 2022
# Functions in this python file generate a randomized mask for SVE based single-shot hdr imaging. They then simulate a
# single SVE masked image by splicing together a set of sequential varied exposure images based on the generated mask.
# This is a part of the SingleShotHDR project (Single-Shot HDR Imaging via Compressed Sensing) submitted in fulfillment
# of the EE367 Computational Imaging Final Project at Stanford University.

import numpy as np


def generate_mask(img_size, num_expsrs):
    # ==================================================================================================================
    # This function generates a randomized grayscale mask for SVE based single-shot hdr imaging with N unique exposures.
    # Inputs:
    #   img_size   - The shape (m x n) of the image/sensor to be masked. [2-element array like]
    #   num_expsrs - The number of unique exposures that will be used. [int]
    #
    # Outputs:
    #   mask       - The grayscale mask with whole-number values corresponding to the kth exposure to be used at each
    #                pixel. Note that minimum value is 1. [ndarray of shape img_size]
    # ==================================================================================================================

    Npix = img_size[0] * img_size[1]  # number of pixels in image
    mask = np.zeros((Npix, ))  # flattened mask
    idx = np.array(range(Npix))  # available indices in mask
    N = int(Npix / num_expsrs)  # number of indices to assign for each mask value

    for k in range(num_expsrs):
        if k + 1 == num_expsrs:
            mask[idx] = k + 1
            continue
        choose = np.random.choice(range(len(idx)), N, replace=False)
        idx_assign = idx[choose]  # get indices for next assignment
        mask[idx_assign] = k + 1  # assign exposure key values

        idx = np.delete(idx, choose)  # remove used indices

    return np.reshape(mask, img_size)


def simulate_SVE_image(image_set, mask):
    # ==================================================================================================================
    # This function simulates a single-shot SVE image from a set of N unique exposure LDR images and a SVE mask of N
    # unique values.
    # Inputs:
    #   image_set - The multi-exposure set of N LDR images. [list of ndarrays]
    #   mask      - The grayscale mask with whole-number values corresponding to the kth exposure to be used at each
    #               pixel. Note that minimum value is 1. [ndarray]
    #
    # Outputs:
    #   sve       - The simulated single-shot SVE image spliced together from the image_set. [ndarray]
    # ==================================================================================================================
    sve = np.zeros(np.shape(image_set[0]))

    for k in range(len(image_set)):
        idx = mask == (k + 1)
        sve[idx] = image_set[k][idx]

    return sve

