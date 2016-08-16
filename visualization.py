import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_velocity_vector(opt_flow, step, trans=0):
    if trans == 0:
        flow = opt_flow
    else:
        flow = opt_flow - trans
    img = np.ones(flow.shape[:2] + (3,))
    for i in range(0, img.shape[0] - step, step):
        for j in range(0, img.shape[1] - step, step):
            try:
                # opencv 3.1.0
                if flow.shape[-1] == 2:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                    (150, 0, 0), 2)
                else:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)

            except AttributeError:
                # opencv 2.4.8
                if flow.shape[-1] == 2:
                    cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                             (150, 0, 0), 2)
                else:
                    cv2.line(img, pt1=(j, i), pt2=(j + int(round(flow[i, j])), i), color=(150, 0, 0), thickness=1)

    plt.figure()
    plt.imshow(img)
    plt.title('velocity vector')
    plt.waitforbuttonpress()


def plot_velocity_vector_mask(opt_flow, mask, step, trans=0):
    if trans == 0:
        flow = opt_flow
    else:
        flow = opt_flow - trans

    img = np.ones(flow.shape[:2] + (3,))
    for i in range(0, img.shape[0] - step, step):
        for j in range(0, img.shape[1] - step, step):
            if mask[i, j] != 0:
                try:
                    # opencv 3.1.0
                    if flow.shape[-1] == 2:
                        cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                        (150, 0, 0), 2)
                    else:
                        cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)

                except AttributeError:
                    # opencv 2.4.8
                    if flow.shape[-1] == 2:
                        cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                 (150, 0, 0), 2)
                    else:
                        cv2.line(img, pt1=(j, i), pt2=(j + int(round(flow[i, j])), i), color=(150, 0, 0), thickness=1)

    plt.figure()
    plt.imshow(img)
    plt.title('velocity vector')
    plt.waitforbuttonpress()


# def point_visualization(x, y, x_gt, y_gt, img):
#     img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # cv2.circle(np.int8(img_show), (np.int8(x), np.int8(y)), 200, (35, 0, 255), -1)
#     fig = plt.figure()
#     plt.imshow(img_show)
#     plt.scatter([x], [y], s=100)  # detected vanishing point, blue dot
#     plt.scatter([x_gt], [y_gt], c='r', s=100)  # ground truth, red dot
#
#     # maximize figure window
#     # manager = plt.get_current_fig_manager()
#     # manager.window.showMaximized()
#     #
#     # plt.ion()
#     # plt.show()
#     # plt.waitforbuttonpress()
#     return fig

def point_visualization(x, y, img):
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.circle(np.int8(img_show), (np.int8(x), np.int8(y)), 200, (35, 0, 255), -1)
    fig = plt.figure()
    plt.imshow(img_show)
    plt.scatter([x], [y], s=100)  # detected vanishing point, blue dot

    # maximize figure window
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    #
    # plt.ion()
    # plt.show()
    # plt.waitforbuttonpress()
    return fig