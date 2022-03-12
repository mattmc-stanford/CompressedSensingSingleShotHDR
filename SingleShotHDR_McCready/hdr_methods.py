# Matthew A. McCready, Department of Electrical Engineering, Stanford University, 2022
# This python file contains functions for performing multi-exposure fusion and tonemapping of hdr images. It is a part
# of the SingleShotHDR project (Single-Shot HDR Imaging via Compressed Sensing) submitted in fulfillment of the EE367
# Computational Imaging Final Project at Stanford University.

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import imageio
import cv2


def multi_expsr_fusion(ldr_set, expsr, filenames):
    # ==================================================================================================================
    # This function calculates the HDR image from a multi-exposure set of LDR images via Debevec et al's MEF technique.
    # It performs a simple scaling and gamma correction tonemapping and Drago tonemapping to save LDR versions of the
    # result. The full HDR result is also saved as a .hdr file.
    # Inputs:
    #   ldr_set   - The multi-exposure LDR image set as a list of images. NOTE: it is assumed that these images have a
    #               simple gamma correction CRF and can be linearized by raising them to the power of 2.2. This can be
    #               easily removed from the code and images linearized before inputting. [list of ndarrays]
    #   expsr     - The exposures in seconds corresponding to the images in ldr_set. [array-like object]
    #   filenames - The filenames in order for the: scaled tonemapped image, Drago tonemapped image, and HDR file. This
    #               is a list of strings WITHOUT the file type identifiers. [list of strings]
    #
    # Outputs:
    #   hdr       - The calculated hdr image. [ndarray]
    # ==================================================================================================================
    wts_set = [None] * len(expsr)  # store ldr weights
    hdr = np.zeros_like(ldr_set[0], dtype=float)
    scale = 0
    for k in range(len(expsr)):
        ldr_set[k] = np.clip(ldr_set[k], 0, 1) ** 2.2 + np.finfo(np.float32).eps
        wts_set[k] = np.exp(-16 * (ldr_set[k] - 0.5) ** 2)

        hdr = hdr + wts_set[k] * (np.log(ldr_set[k]) - np.log(expsr[k]))
        scale = scale + wts_set[k]

    # Normalize
    hdr = np.exp(hdr / scale)
    hdr *= 0.8371896 / np.mean(hdr)  # makes mean of created HDR image match reference image (totally optional)

    # convert to 32 bit floating point format, required for OpenCV
    hdr = np.float32(hdr)

    # make and save our tonemapping
    s = 0.7
    g = 0.5
    ldrMe = (s * hdr) ** g
    io.imsave(filenames[0] + '.jpg', np.uint8(np.clip(ldrMe, 0, 1) * 255))

    # make and save Drago tonemapping
    gamma = 1.0  # 1.0
    saturation = 0.5  # 0.7
    bias = 0.85  # 0.85
    tonemapDrago = cv2.createTonemapDrago(gamma, saturation, bias)
    ldrDrago = tonemapDrago.process(hdr)
    io.imsave(filenames[1] + '.jpg', np.uint8(np.clip(3 * ldrDrago, 0, 1) * 255))

    # write HDR image (can compare to hw4_1_memorial_church.hdr reference image in an external viewer)
    hdrCV = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filenames[2] + '.hdr', hdrCV)

    return hdr






