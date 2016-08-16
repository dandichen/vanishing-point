# super pixel methods
import numpy as np
import cv2
import sys
import time

__author__ = "haoyi liang"


class Superpixel(object):
    def __init__(self, ite_num=10):
        self.ite_num = ite_num

    def seeds(self, img, num_superpixels=200, num_levels=4, prior=2, num_histogram_bins=5):
        start_time = time.time()
        converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width, channels = converted_img.shape
        seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels,
                                                   prior, num_histogram_bins)
        seeds.iterate(converted_img, self.ite_num)
        sp_num = seeds.getNumberOfSuperpixels()
        labels = seeds.getLabels()
        mask = seeds.getLabelContourMask(False)
        run_time = time.time() - start_time
        return mask, labels, sp_num, run_time

    def lsc(self, img, min_element_size=100):
        start_time = time.time()
        lsc_obj = cv2.ximgproc.createSuperpixelLSC(img)
        lsc_obj.iterate(self.ite_num)
        lsc_obj.enforceLabelConnectivity(min_element_size)
        labels = lsc_obj.getLabels()
        sp_num = lsc_obj.getNumberOfSuperpixels()
        mask = lsc_obj.getLabelContourMask(False)
        run_time = time.time() - start_time
        return mask, labels, sp_num, run_time

    def slic(self, img, min_element_size=100):
        start_time = time.time()
        slic_obj = cv2.ximgproc.createSuperpixelSLIC(img)
        slic_obj.iterate(self.ite_num)
        slic_obj.enforceLabelConnectivity(min_element_size)
        labels = slic_obj.getLabels()
        sp_num = slic_obj.getNumberOfSuperpixels()
        mask = slic_obj.getLabelContourMask(False)
        run_time = time.time() - start_time
        return mask, labels, sp_num, run_time
