import cv2
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'haoyi liang'


def temp_demo(corner_points, img0, opt_temp):
    corner_points = np.uint16(corner_points)
    opt_temp = np.int16(opt_temp)
    mask = np.zeros_like(img0)
    sparse_corner_num, _ = corner_points.shape
    color = np.random.randint(0, 255, (sparse_corner_num, 3))
    length_demo_ratio = 200.0 / np.max(np.abs(opt_temp))
    for i, (u, v) in enumerate(corner_points):
            [opt_x, opt_y] = np.int16(opt_temp[i] * length_demo_ratio)
            mask = cv2.line(mask, (u, v), (u + opt_x, v + opt_y), color[i].tolist(), 2)
            img0 = cv2.circle(img0, (u, v), 5, color[i].tolist(), -1)
            img = cv2.add(img0, mask)
    plt.figure()
    plt.imshow(img)
    plt.title('flow template')
    plt.waitforbuttonpress()


def sparse_opt_demo(corner_points, img0, sparse_opt):
    corner_points = np.uint16(corner_points)
    sparse_opt = np.int16(sparse_opt)
    mask = np.zeros_like(img0)
    sparse_corner_num, _ = corner_points.shape
    color = np.random.randint(0, 255, (sparse_corner_num, 3))
    length_demo_ratio = 200.0/np.max(np.abs(sparse_opt))
    for i, (u, v) in enumerate(corner_points):
        [opt_x, opt_y] = np.int16(sparse_opt[i] * length_demo_ratio)
        mask = cv2.line(mask, (u, v), (u + opt_x, v + opt_y), color[i].tolist(), 2)
        img0 = cv2.circle(img0, (u, v), 5, color[i].tolist(), -1)
        img = cv2.add(img0, mask)
    plt.figure()
    plt.imshow(img)
    plt.title('original flow')
    plt.waitforbuttonpress()


def depth_demo(corner_points, img0, depth):
    corner_points = np.uint16(corner_points)
    depth = np.int16(depth)
    mask = np.zeros_like(img0)
    sparse_corner_num, _ = corner_points.shape
    color = np.random.randint(0, 255, (sparse_corner_num, 3))
    length_demo_ratio = 200.0 / np.max(np.abs(depth))
    for i, (u, v) in enumerate(corner_points):
        depth_num = np.int16(depth[i]*length_demo_ratio)
        mask = cv2.line(mask, (u, v), (u, v + depth_num), color[i].tolist(), 2)
        img0 = cv2.circle(img0, (u, v), 5, color[i].tolist(), -1)
        img = cv2.add(img0, mask)
    plt.figure()
    plt.imshow(img)
    plt.title('depth')
    plt.waitforbuttonpress()


def confidence_demo(corner_points, img0, flow_confidence):
    corner_points = np.uint16(corner_points)
    flow_confidence = np.int16(flow_confidence)
    mask = np.zeros_like(img0)
    sparse_corner_num, _ = corner_points.shape
    color = np.random.randint(0, 255, (sparse_corner_num, 3))
    length_demo_ratio = 200.0 / np.max(np.abs(flow_confidence))
    for i, (u, v) in enumerate(corner_points):
        conf = np.int16(flow_confidence[i]*length_demo_ratio)
        mask = cv2.line(mask, (u, v), (u, v + conf), color[i].tolist(), 2)
        img0 = cv2.circle(img0, (u, v), 5, color[i].tolist(), -1)
        img = cv2.add(img0, mask)
    plt.figure()
    plt.imshow(img)
    plt.title('confidence')
    plt.waitforbuttonpress()


def moving_points_demo(corner_points, img0, is_static_points, angle_dif):
    corner_points = np.uint16(corner_points)
    is_static_points = np.int16(is_static_points)
    mask0 = np.zeros_like(img0)
    sparse_corner_num, _ = corner_points.shape
    color = np.random.randint(0, 255, (sparse_corner_num, 3))
    for i, (u, v) in enumerate(corner_points):
        flag_dir = np.int8((is_static_points[i] - 0.5) * 100)
        mask0 = cv2.line(mask0, (u, v), (u, v + flag_dir), color[i].tolist(), 2)
        img0 = cv2.circle(img0, (u, v), 5, color[i].tolist(), -1)
        img = cv2.add(img0, mask0)
    plt.figure()
    plt.imshow(img)
    plt.title('if static')
    plt.waitforbuttonpress()

    mask1 = np.zeros_like(img0)
    sparse_corner_num, _ = corner_points.shape
    color = np.random.randint(0, 255, (sparse_corner_num, 3))
    for i, (u, v) in enumerate(corner_points):
        angle_dif_num = np.int16((angle_dif[i]))
        mask1 = cv2.line(mask1, (u, v), (u, v + angle_dif_num), color[i].tolist(), 2)
        img0 = cv2.circle(img0, (u, v), 5, color[i].tolist(), -1)
        img = cv2.add(img0, mask1)
    plt.figure()
    plt.title('angle diff')
    plt.imshow(img)

    plt.waitforbuttonpress()


def v_points_demo(vote_map):
    plt.figure()
    plt.plot(vote_map[:, 0])
    plt.title('voting: col vs. row')
    plt.waitforbuttonpress()
    plt.figure()
    plt.plot(vote_map[:, 1])
    plt.title('voting: col confidence')
    plt.waitforbuttonpress()
