# dense flow using super pixel
import cv2
import numpy as np
__author__ = "Dandi Chen"


def get_superpixel(img, num_superpixel=200):
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, channels = converted_img.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixel, num_levels=4,
                                               prior=2, histogram_bins=5)
    seeds.iterate(converted_img, 10)
    sp_num = seeds.getNumberOfSuperpixels()
    labels = seeds.getLabels()
    mask = seeds.getLabelContourMask(False)
    return sp_num, labels, mask


def dense2sparse(img_prev, img_next, num_superpixel, labels, pyr_scale=0.5,
                 levels=3, winsize=15, iterations=10, poly_n=5, poly_sigma=1.2, flags=0):
    imgPrev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    imgNext = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(imgPrev, imgNext, None, pyr_scale, levels,
                                        winsize, iterations, poly_n, poly_sigma, flags)

    flow_X = flow[:, :, 0]
    flow_Y = flow[:, :, 1]

    arr = []
    for label_index in range(num_superpixel):
        flow_X_mean = np.mean(flow_X[np.where(labels == label_index)])
        flow_Y_mean = np.mean(flow_Y[np.where(labels == label_index)])

        x, y = np.where(labels == label_index)
        x_mean = round(np.mean(x))
        y_mean = round(np.mean(y))

        arr.append([x_mean, y_mean, flow_X_mean, flow_Y_mean])
    return arr