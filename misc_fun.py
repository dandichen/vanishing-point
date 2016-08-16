import numpy as np
import denseflow
import os


def mxnet_flow(img0, sparse_corner_num, num):
    num_superpixel, labels, _ = denseflow.get_superpixel(img0, num_superpixel=sparse_corner_num)
    # flowstereo(mxnet)
    flow_path = os.path.join('/mnt/scratch/DandiChen/road/flow/6/flowstereo/value', '{:06d}'.format(num))
    # opencv dense flow
    # flow_path = os.path.join('/mnt/scratch/DandiChen/road/flow/1/opencv_dense/value', '{:06d}'.format(num))
    # opencv sparse flow
    # flow_path = os.path.join('/mnt/scratch/DandiChen/road/flow/1/opencv_sparse/value', '{:06d}'.format(num))

    flow = np.load(flow_path+'.npz')
    flow_x = flow['flow_X']
    flow_y = flow['flow_Y']
    arr = []
    for label_index in range(num_superpixel):
        flow_x_mean = np.mean(flow_x[np.where(labels == label_index)])
        flow_y_mean = np.mean(flow_y[np.where(labels == label_index)])

        x, y = np.where(labels == label_index)
        x_mean = round(np.mean(x))
        y_mean = round(np.mean(y))

        arr.append([x_mean, y_mean, flow_x_mean, flow_y_mean])
    results = np.array(arr)
    sp_pos = np.array([results[:, 1], results[:, 0]]).transpose()
    sp_flow = results[:, 2:4]
    return sp_pos, sp_flow, img0
