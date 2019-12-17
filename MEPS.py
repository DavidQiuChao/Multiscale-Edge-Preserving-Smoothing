#!/usr/bin/env python
#coding=utf-8

import os
import cv2
import numpy as np
from scipy import misc,ndimage
import matplotlib.pyplot as plt


def guidedDownsamp(gd,src,r,eps):
    gd=  gd.astype(np.float32)
    src=src.astype(np.float32)
    res = cv2.ximgproc.guidedFilter(gd,src,r,eps)
    return res[::2,::2]


def get_guided_pyramid(W,L,n):
    guided_pyramid = [W]
    for floor in range(1,n):
        r = int((n-floor)*2)
        eps = (n-floor)**0.5
        W = guidedDownsamp(L,W,r,eps)
        L = L[::2,::2]
        guided_pyramid.append(W)
    return guided_pyramid


def show(color_array):
    """ Function to show image"""
    plt.imshow(color_array)
    plt.show()
    plt.axis('off')


def exponential_euclidean(canal, sigma):
    return np.exp(-(canal - 0.5)**2 / (2 * sigma**2))


def avgFilter(srcIm,k=31):
    dstIm = ndimage.uniform_filter(srcIm,size=k)
    return dstIm


class lGImage():
    def __init__(self,path):
        rgb = misc.imread(path)
        self.rgb = rgb.astype(np.float32)/255.
        self.shape = self.rgb.shape
        self.L = np.dot(self.rgb[...,:3],[0.299,0.587,0.114])

    def localGlobalWeights(self):
        # local weights
        B = avgFilter(self.L,31)
        wl = exponential_euclidean(B,0.1)
        meanL = np.mean(self.L)
        wg = exponential_euclidean(meanL,0.5)
        return wl*wg
