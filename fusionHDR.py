#!/usr/bin/env python
#coding-utf-8


import imageio
import logging
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import utils
import MEPS
from MEPS import get_guided_pyramid
from MEPS import lGImage

logging.getLogger().setLevel(logging.INFO)


class HDRFusion(object):
    """Class for weights attribution with Laplacian Fusion"""

    def __init__(self, names):
        self.images = []
        for name in names:
            self.images.append(lGImage(name))
        self.shape = self.images[0].shape
        self.num_images = len(self.images)
        #height of pyramid
        self.T = np.log2(min(self.shape[0],\
                self.shape[1]))
        self.T = int(np.ceil(self.T))

    def get_init_weights_map(self):
        """Return the normalized Weight map"""
        self.weights = []
        sums = np.zeros((self.shape[0], self.shape[1]))
        for image_name in self.images:
            weight = image_name.localGlobalWeights()+1e-12
            self.weights.append(weight)
            sums = sums + weight
        for index in range(self.num_images):
            self.weights[index] = self.weights[index] / sums
        return self.weights

    def get_gaussian_pyramid(self, image, n):
        """Return the Gaussian Pyramid of an image"""
        gaussian_pyramid_floors = [image]
        for floor in range(1, n):
            gaussian_pyramid_floors.append(
                utils.Reduce(gaussian_pyramid_floors[-1], 1))
        return gaussian_pyramid_floors

    def get_guided_pyramid_weights(self):
        """Return the Guided Pyramid of the Weight map of all images"""
        self.weights_pyramid = []
        for index in range(self.num_images):
            self.weights_pyramid.append(\
                get_guided_pyramid(self.weights[index],self.images[index].L,self.T))
        return self.weights_pyramid

    def get_laplacian_pyramid(self, image, n):
        """Return the Laplacian Pyramid of an image"""
        gaussian_pyramid_floors = self.get_gaussian_pyramid(image, n)
        laplacian_pyramid_floors = [gaussian_pyramid_floors[-1]]
        for floor in range(n - 2, -1, -1):
            size = gaussian_pyramid_floors[floor].shape
            new_floor = gaussian_pyramid_floors[floor] - utils.Expand(
                gaussian_pyramid_floors[floor + 1], 1,size)
            laplacian_pyramid_floors = [new_floor] + laplacian_pyramid_floors
        return laplacian_pyramid_floors

    def get_laplacian_pyramid_images(self):
        """Return all the Laplacian pyramid for all images"""
        self.laplacian_pyramid = []
        for index in range(self.num_images):
            self.laplacian_pyramid.append(\
                self.get_laplacian_pyramid(self.images[index].rgb,\
                self.T)
                )
        return self.laplacian_pyramid

    def result_exposure(self):
        "Return the Exposure Fusion image with Laplacian/Gaussian Fusion method"
        print("weights")
        self.get_init_weights_map()
        print ("guided pyramid")
        self.get_guided_pyramid_weights()
        print ("laplacian pyramid")
        self.get_laplacian_pyramid_images()
        print ("do images fusion")
        result_pyramid = []
        for floor in range(self.T):
            #print ('floor ', floor)
            result_floor = np.zeros(self.laplacian_pyramid[0][floor].shape)
            for index in range(self.num_images):
                #print ('image ', index)
                for canal in range(3):
                    result_floor[:, :,canal] += \
                        self.laplacian_pyramid[index][floor][:, :,canal] \
                        * self.weights_pyramid[index][floor]
            result_pyramid.append(result_floor)
        print ("resconstuct HDR image")
        # Get the image from the Laplacian pyramid
        self.result_image = result_pyramid[-1]
        for floor in range(self.T - 2, -1, -1):
            #print ('floor ', floor)
            size = result_pyramid[floor].shape
            self.result_image = result_pyramid[floor] + utils.Expand(
                self.result_image, 1,size)
        self.result_image[self.result_image < 0] = 0
        self.result_image[self.result_image > 1] = 1
        print ("done")
        return self.result_image


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list',help='input image sequence list')
    parser.add_argument('-r','--res',help='result image name')
    args = parser.parse_args()
    lpath = args.list
    oname = args.res
    names = [line.rstrip('\n') for line in open(lpath)]
    start = time.time()
    fsn = HDRFusion(names)
    hdr = fsn.result_exposure()
    end = time.time()
    cost = (end-start)
    print('time cost: {}'.format(cost))
    MEPS.show(hdr)
    imageio.imwrite(oname,hdr)
