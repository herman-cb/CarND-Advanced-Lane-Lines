import cv2
import numpy as np
from calibrate_camera import *

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise Error("Unsupported direction for gradient")
    # Apply threshold
    abs_sobel = np.absolute(sobel)
    abs_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(abs_sobel)
    grad_binary[(abs_sobel >= thresh[0]) & (abs_sobel <= thresh[1])] = 1
    return grad_binary


def hls_thresh(img, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_output = np.zeros_like(s_channel)
    s_output[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_output = np.zeros_like(v_channel)
    v_output[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_output == 1) & (v_output == 1)] = 1

    return output


def window_mask(w, h, img_ref, center, level):
    out = np.zeros_like(img_ref)
    out[int(img_ref.shape[0] - (level + 1) * h):int(img_ref.shape[0] - level * h), max(0, int(center - w)):min(int(center + w), img_ref.shape[1])] = 1
    return out

def warp_image(img):
    img_size = (img.shape[1], img.shape[0])
    bot_width = 0.76
    mid_width = 0.08
    height_pct = 0.62
    bottom_trim = 0.935
    src = np.float32([[img.shape[1] * (0.5 - mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (0.5 + mid_width / 2), img.shape[0] * height_pct],
                      [img.shape[1] * (0.5 + bot_width / 2), img.shape[0] - bottom_trim],
                      [img.shape[1] * (0.5 - bot_width / 2), img.shape[0] - bottom_trim]])
    offset = img_size[0] * 0.25
    # offset = 0
    dst = np.float32([[offset, 0],
                      [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]],
                      [offset, img_size[1]]])
    # print("src = {}".format(src))
    # print("dst = {}".format(dst))
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv


if __name__ == "__main__":
    img = cv2.imread("./harder_frame.png")
    camera_calibration = pickle.load(open("camera_cal.pkl", "rb"))
    undistorted_img = cv2.undistort(img, camera_calibration.mtx, camera_calibration.dist)
    cv2.imwrite("./output_images/harder_frame_undistorted.jpg", undistorted_img)

    # Threshold
    binary_img = np.zeros_like(undistorted_img[:, :, 0])
    gradx = abs_sobel_thresh(undistorted_img, orient='x', thresh=(12, 255))
    grady = abs_sobel_thresh(undistorted_img, orient='y', thresh=(25, 255))
    c_binary = hls_thresh(undistorted_img, sthresh=(100, 255), vthresh=(50, 255))
    binary_img[(gradx == 1) & (grady == 1) | (c_binary == 1)] = 255
    cv2.imwrite("./output_images/harder_frame_binarized.jpg", binary_img)

    #Warp
    warped_image, Minv = warp_image(undistorted_img)
    cv2.imwrite("./output_images/test1_warped.jpg", warped_image)

