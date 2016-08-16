import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import demo_func
import denseflow
import misc_fun

__author__ = 'haoyi liang'


class HidVarInf(object):
    def __init__(self, sparse_corner_num=100, resize_ratio=1, moving_angle_thresh=[30, 5],
                 stop_thresh=1, ite_num=5, is_sparse=0,demo=False):
        self.sparse_corner_num = sparse_corner_num
        self.resize_ratio = resize_ratio
        self.moving_angle_thresh = moving_angle_thresh
        self.stop_thresh = stop_thresh
        self.ite_num = ite_num
        self.demo = demo
        self.is_sparse = is_sparse
        self.motion_status = []

    def get_motion(self, img0, img1, num):
        thresh_angle_log = np.log(self.moving_angle_thresh)
        moving_thresh_angle_list = np.logspace(thresh_angle_log[0], thresh_angle_log[1], self.ite_num, base=np.exp(1))
        height, width, _ = img0.shape
        search_len = width/2
        # if self.is_sparse:
        #     corner_points, opt_flow, img0 = HidVarInf._get_sparse_flow(self, img0, img1, self.sparse_corner_num)
        # else:
        #     corner_points, opt_flow, img0 = HidVarInf._get_dense_flow(self, img0, img1, self.sparse_corner_num)
        corner_points, opt_flow, img0 = misc_fun.mxnet_flow(img0, self.sparse_corner_num, num)
        org_point = [width/2, height/2]
        flow_temp, v_point, depth, flow_confidence, is_static_points = \
            HidVarInf._initial_vars(self, corner_points, opt_flow, height, width, org_point)
        cost_hist = []
        # TODO: replace the terminate condistion by a cost threshold
        for i in range(self.ite_num):
            if self.demo:
                demo_func.sparse_opt_demo(corner_points, img0, opt_flow)

            v_point = HidVarInf._upd_v_point(self, corner_points, opt_flow, is_static_points, flow_confidence, height, width)
            opt_temp = HidVarInf._upd_opt_temp(self, v_point, corner_points, img0, depth)
            depth = HidVarInf._upd_depth(self, opt_temp, opt_flow, corner_points, img0)
            is_static_points = HidVarInf._upd_moving_points(self, opt_temp, opt_flow, moving_thresh_angle_list[i], corner_points, img0)
            flow_confidence = HidVarInf._upd_confidence(self, opt_temp, opt_flow, is_static_points, corner_points, img0,
                                                      moving_thresh_angle_list[i])
            cur_cost = HidVarInf._evaluate_v_point(self, opt_temp, opt_flow, is_static_points, moving_thresh_angle_list[i])
            cost_hist.append(cur_cost)
            if self.demo:
                print('updated vp: {} and {}'.format(v_point[0], v_point[1]))
                plt.close('all')
        vel = HidVarInf._get_vel(self, corner_points, opt_flow, v_point, is_static_points)
        if HidVarInf._is_stop(self, corner_points, opt_flow, v_point, self.stop_thresh, height, width):
            # v_point = org_point
            # vel = 0
            is_stop = 1
        else:
            is_stop = 0
        return v_point, vel, is_stop, cost_hist[self.ite_num-1]

    def _upd_v_point(self, corner_points, opt_flow, is_static_points, flow_confidence, height, width):
        opt_flow = -opt_flow
        corner_nums = len(corner_points)
        weight = flow_confidence
        points_x_coords = corner_points[:, 0].reshape(corner_nums, 1)
        points_y_coords = corner_points[:, 1].reshape(corner_nums, 1)
        opt_flow_x = opt_flow[:, 0].reshape(corner_nums, 1)
        opt_flow_y = opt_flow[:, 1].reshape(corner_nums, 1)
        vote_range = width / 2 + np.tan(np.linspace(-3.14 / 2, 3.14 / 2, 180)) * width / 2
        vote_map = np.zeros((180, 2))

        for line_ind, line_num in enumerate(vote_range):
            row_num = (line_num - points_x_coords) / opt_flow_x * opt_flow_y + points_y_coords
            nan_inf_pos = np.float16(np.logical_or(np.isnan(row_num), np.isinf(row_num)))
            right_dir = np.float16((np.sign(opt_flow_x) * np.sign(line_num - points_x_coords)) > 0)
            row_num[np.where(nan_inf_pos)] = 0

            valid_vote = is_static_points * (1 - nan_inf_pos) * right_dir
            valid_vote[np.where(row_num < 0)] = 0
            valid_vote[np.where(row_num > height)] = 0
            valid_weighted_row_num = row_num * weight * valid_vote

            row_num_mean = np.int16(np.mean(valid_weighted_row_num) / np.mean(weight * valid_vote))
            if np.sum(valid_vote) > 5:
                row_conf = np.std(row_num[np.where(valid_vote)]) / (np.sum(valid_vote) ** 2)
            else:
                row_conf = np.inf
            vote_map[line_ind, :] = [row_num_mean, row_conf]
        v_point_ind = np.argmin(vote_map[:, 1])
        v_point = np.array([vote_range[v_point_ind], vote_map[v_point_ind, 0]])

        if self.demo:
            demo_func.v_points_demo(vote_map)
        return v_point

    def _upd_opt_temp(self, v_point, corner_points, img0, depth):
        opt_temp = (corner_points - v_point) / depth
        if self.demo:
            demo_func.temp_demo(corner_points, img0, opt_temp)
        return opt_temp

    def _upd_depth(self, opt_temp, opt_flow, corner_points, img0):
        opt_temp_amp = (opt_temp[:, 0] ** 2 + opt_temp[:, 1] ** 2) ** 0.5
        opt_flow_amp = (opt_flow[:, 0] ** 2 + opt_flow[:, 1] ** 2) ** 0.5
        corner_num = len(opt_temp)
        depth = opt_temp_amp / opt_flow_amp
        depth = depth.reshape(corner_num, 1)
        if self.demo:
            demo_func.depth_demo(corner_points, img0, depth)
        return depth

    def _upd_moving_points(self, opt_temp, opt_flow, moving_thresh_angle, corner_points, img0):
        opt_temp_amp = (opt_temp[:, 0] ** 2 + opt_temp[:, 1] ** 2) ** 0.5
        opt_flow_amp = (opt_flow[:, 0] ** 2 + opt_flow[:, 1] ** 2) ** 0.5
        angle_dif = np.arccos(np.sum(opt_temp * opt_flow, axis=1) / (opt_temp_amp * opt_flow_amp)) / 3.14 * 180
        corner_num = len(corner_points)
        is_static_points = np.float16(angle_dif < moving_thresh_angle).reshape(corner_num, 1)
        if self.demo:
            demo_func.moving_points_demo(corner_points, img0, is_static_points, angle_dif)
        return is_static_points

    def _upd_confidence(self, opt_temp, opt_flow, is_static_points, corner_points, img0, moving_thresh_angle):
        # TODO: optical flows that are more "vertical" should be asigned more weight since they are better at locating v point
        corner_num, _ = opt_flow.shape
        flow_amp = (opt_flow[:, 0] ** 2 + opt_flow[:, 1] ** 2) ** 0.5
        flow_amp = flow_amp.reshape(corner_num, 1)
        temp_amp = (opt_temp[:, 0] ** 2 + opt_temp[:, 1] ** 2) ** 0.5
        temp_amp = temp_amp.reshape(corner_num, 1)
        angle_diff = np.arccos(
            np.sum(opt_temp * opt_flow, axis=1).reshape(corner_num, 1) / (temp_amp * flow_amp)) / 3.14 * 180
        angle_confidence = moving_thresh_angle - angle_diff
        angle_confidence[np.where(angle_confidence < 0)] = 0
        flow_confidence = flow_amp * is_static_points * angle_confidence
        flow_confidence[np.where(np.isnan(flow_confidence))] = 0
        if self.demo:
            demo_func.confidence_demo(corner_points, img0, flow_confidence)

        return flow_confidence

    def _initial_vars(self, corner_points, opt_flow, height, width, org_point):
        points_num = len(corner_points)
        corner_points = np.array(corner_points)
        v_point = np.array([org_point[1], org_point[0]])
        flow_confidence = np.ones((points_num, 1))
        is_static_points = np.ones((points_num, 1))
        flow_temp = np.zeros((points_num, 2))
        flow_temp[:, 0] = corner_points[:, 0] - width / 2
        flow_temp[:, 1] = corner_points[:, 1] - height / 2
        temp_amp = (flow_temp[:, 0] ** 2 + flow_temp[:, 1] ** 2) ** 0.5
        opt_amp = (opt_flow[:, 0] ** 2 + opt_flow[:, 1] ** 2) ** 0.5
        depth = (temp_amp / opt_amp).reshape(points_num, 1)
        return flow_temp, v_point, depth, flow_confidence, is_static_points

    def _get_dense_flow(self, img0, img1, num):
        num_superpixel, labels, _ = denseflow.get_superpixel(img0, num_superpixel=num)
        results = denseflow.dense2sparse(img0, img1, num_superpixel, labels)
        results = np.array(results)
        sp_pos = np.array([results[:, 1], results[:, 0]]).transpose()
        sp_flow = results[:, 2:4]
        return sp_pos, sp_flow, img0

    def _get_sparse_flow(self, img0, img1, sparse_corner_num):
        img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # TODO: sometimes p0 is none
        p0 = cv2.goodFeaturesToTrack(img0_gray, mask=None, maxCorners=sparse_corner_num, qualityLevel=0.3,
                                     minDistance=7, blockSize=7)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0_gray, img1_gray, p0, None, winSize=(15, 15), maxLevel=2,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        opt_flow = good_new - good_old
        return good_old, opt_flow, img0

    def _evaluate_v_point(self, opt_temp, opt_flow, is_static_points, moving_thresh_angle):
        corner_num = len(opt_temp)
        fit_per = np.float16(np.sum(is_static_points)) / corner_num
        opt_temp_amp = (opt_temp[:, 0] ** 2 + opt_temp[:, 1] ** 2) ** 0.5
        opt_flow_amp = (opt_flow[:, 0] ** 2 + opt_flow[:, 1] ** 2) ** 0.5
        angle_dif = np.arccos(np.sum(opt_temp * opt_flow, axis=1) / (opt_temp_amp * opt_flow_amp)) / 3.14 * 180
        fit_acc = np.mean(angle_dif[np.where(angle_dif < moving_thresh_angle)])
        cost = np.array([fit_per, fit_acc])
        return cost

    def _get_vel(self, corner_points, opt_flow, v_point, is_static_points):
        corner_num = len(corner_points)
        flow_dis = (((corner_points[:, 0] - v_point[0]) ** 2 + (corner_points[:, 1] - v_point[1]) ** 2) ** 0.5).reshape(
            corner_num, 1)
        flow_amp = ((opt_flow[:, 0] ** 2 + opt_flow[:, 1] ** 2) ** 0.5).reshape(corner_num, 1)
        vel_list = flow_amp / flow_dis
        vel = np.mean(vel_list[np.where(is_static_points)])
        return vel

    def _is_stop(self, corner_points, opt_flow, v_point, stop_thresh, height, width):
        corner_num = len(opt_flow)
        diag_len = (height ** 2 + width ** 2) ** 0.5
        opt_flow_amp = ((opt_flow[:, 0] ** 2 + opt_flow[:, 1] ** 2) ** 0.5).reshape((corner_num, 1))
        relative_stop_pt = np.where(opt_flow_amp < stop_thresh)
        relative_stop_pt_pos = corner_points[relative_stop_pt[0], :] - v_point
        relative_stop_pt_dis = (relative_stop_pt_pos[:, 0] ** 2 + relative_stop_pt_pos[:, 1] ** 2) ** 0.5
        if (np.median(relative_stop_pt_dis) > diag_len * 0.1) & (
            np.sum(opt_flow_amp < stop_thresh) > 2 * corner_num / 3):
            if_stop = True
        else:
            if_stop = False
        return if_stop