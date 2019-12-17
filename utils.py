#!/usr/bin/env python
#coding=utf-8

import numpy as np
import scipy.signal as sig


def kernel_1D(n, a=0.6):
    """Kernel function in 1 dimension"""
    kernel = [.0625, .25, .375, .25, .0625]
    return kernel[n]


def get_kernel(a=0.6):
    kernel = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            kernel[i, j] = kernel_1D(i, a)*kernel_1D(j, a)
    return kernel


def Reduce1(image, a=0.6):
    kernel = get_kernel(a)
    shape = image.shape
    if len(shape) == 3:
        image_reduced = np.zeros((int(np.ceil(shape[0]/2)),\
                int(np.ceil(shape[1]/2)),3))
        for canal in range(3):
            canal_reduced = sig.convolve2d(image[:, :, canal], kernel, 'same')
            image_reduced[:, :, canal] = canal_reduced[::2, ::2]
    else:
        image_reduced = sig.convolve2d(image, kernel, 'same')[::2, ::2]
    return image_reduced


def Reduce(image, n, a=0.6):
    """Reduce function for Pyramids"""
    try:
        if n == 0:
            return image
        else:
            image = Reduce(image, n-1, a)
            return Reduce1(image, a)
    except Exception as e:
        print ("Dimension Error")
        print (e)


def Expand1(image, size, a=0.6):
    kernel = get_kernel(a)
    shape = image.shape
    if len(shape) == 3:
        image_to_expand = np.zeros((size[0], size[1], 3))
        image_expanded = np.zeros(image_to_expand.shape)
        for canal in range(3):
            image_to_expand[::2, ::2, canal] = image[:, :, canal]
            image_expanded[:, :, canal] = sig.convolve2d(image_to_expand[:, :, canal], 4*kernel, 'same')
    else:
        image_to_expand = np.zeros((size[0], shape[1]))
        image_to_expand[::2, ::2] = image
        image_expanded = sig.convolve2d(image_to_expand[:, :], 4*kernel, 'same')
    return image_expanded


def Expand(image, n, size,a=0.6):
    """Expand function for Pyramids"""
    try:
        if n == 0:
            return image
        else:
            image = Expand(image, n-1, size,a)
            return Expand1(image, size,a)
    except Exception as e:
        print ("Dimension error")
        print (e)
