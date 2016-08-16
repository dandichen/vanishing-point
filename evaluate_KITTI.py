import sys
sys.path.insert(0, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib/python')
sys.path.insert(1, '/mnt/scratch/third-party-packages/libopencv_3.1.0/lib')

import numpy as np
from hid_inference import HidVarInf
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from scipy import stats
import visualization as vis

print cv2.__version__

vis_path = '/mnt/scratch/DandiChen/road/vanishing_point/3/opencv_sparse/visualization'

video2img_path = '/mnt/scratch/DandiChen/road/data/3'
img_num = len([name for name in os.listdir(video2img_path) if os.path.isfile(os.path.join(video2img_path, name))]) - 1
flow_num = img_num - 1  # continuous two frames

def read_img(gt_path, img_num):
    img0 = cv2.imread(os.path.join(video2img_path, str(img_num).zfill(6) + '.png'))
    img1 = cv2.imread(os.path.join(video2img_path, str(img_num + 1).zfill(6) + '.png'))

    # img0_name = os.path.join(gt_path, str(img_num).zfill(6) + '_10.png')
    # img1_name = os.path.join(gt_path, str(img_num).zfill(6) + '_11.png')
    # img0 = cv2.imread(img0_name)
    # img1 = cv2.imread(img1_name)
    return img0, img1

def main():
    # gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2'
    # row_ana_gt_path = '/mnt/scratch/haoyiliang/todandi/extracted_data'
    demo = False
    is_sparse = 1
    test_segment = np.arange(0, flow_num - 1)
    org_point = np.array([610, 173])
    sparse_corner_num = 150
    resize_ratio = 1
    moving_angle_thresh = [30, 10]
    stop_thresh = 1
    ite_num = 5
    v_point = np.zeros((len(test_segment), 2))
    cost = np.zeros((len(test_segment), 2))
    vel = np.zeros((len(test_segment), 1))
    is_stop = np.zeros((len(test_segment), 1))
    # row_gt_v_point = np.zeros((len(test_segment), 2))
    # row_gt_vel = np.zeros((len(test_segment), 1))
    infer = HidVarInf(sparse_corner_num=sparse_corner_num, resize_ratio=resize_ratio, moving_angle_thresh=moving_angle_thresh,
                      stop_thresh=stop_thresh, ite_num=ite_num, is_sparse=is_sparse, demo=demo)
    for ind, img_num in enumerate(test_segment):
        print('image number is {}'.format(img_num))
        # row_gt_data = np.load(os.path.join(row_ana_gt_path, '{:06d}'.format(img_num) + '.npz'))
        # row_gt_v_point[ind, :] = row_gt_data['v_point']
        # row_gt_vel[ind] = row_gt_data['ego_vel']
        # print('the ground truth vanishing point is {} and {}'.format(row_gt_v_point[ind, 0], row_gt_v_point[ind, 1]))
        # print("the ground truth direction is {}".format((np.arctan((row_gt_v_point[ind, 0] - org_point[0]) / org_point[0])) / 3.14 * 180))

        img0, img1 = read_img(video2img_path, img_num)
        v_point[ind, :], vel[ind], is_stop[ind], cost[ind, :] = infer.get_motion(img0, img1, img_num)
        print('the detected vanishing point is {} and {}'.format(v_point[ind, 0], v_point[ind, 1]))
        print("the detected direction is {}".format((np.arctan((v_point[ind, 0] - org_point[0]) / org_point[0])) / 3.14 * 180))

        # vanishing point visualization
        # fig = vis.point_visualization(v_point[ind, 0], v_point[ind, 1], row_gt_v_point[ind, 0], row_gt_v_point[ind, 1], img0)
        fig = vis.point_visualization(v_point[ind, 0], v_point[ind, 1], img0)
        fig.savefig(os.path.join(vis_path, str(img_num).zfill(6) + '.png'))
        # plt.imsave(os.path.join(vis_path + '/test', str(img_num).zfill(6) + '.png'), fig)

    # gt_dir = np.arctan((row_gt_v_point[:, 0] - org_point[0]) / org_point[0]) / 3.14 * 180
    # plt.plot(gt_dir)
    # plt.title('groundtruth direction')
    # plt.waitforbuttonpress()
    # plt.figure()
    # dect_dir = np.arctan((v_point[:, 0] - org_point[0]) / org_point[0]) / 3.14 * 180
    # plt.plot(dect_dir)
    # plt.title('detected direction')
    # plt.waitforbuttonpress()
    # dir_linear_cor = stats.pearsonr(dect_dir, gt_dir)
    # print('correlation between ground truth and detection is {}'.format(dir_linear_cor))
    #
    # plt.figure()
    # plt.plot(row_gt_vel)
    # plt.title('groundtruth velocity')
    # plt.waitforbuttonpress()
    # plt.figure()
    # plt.plot(vel)
    # plt.title('detected velocity')
    # plt.waitforbuttonpress()
    # vel_linear_cor = stats.pearsonr(vel, row_gt_vel)
    # print('correlation between ground truth and detection is {}'.format(vel_linear_cor))

if __name__ == '__main__':
    main()

