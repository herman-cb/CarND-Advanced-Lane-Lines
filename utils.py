import cv2
import numpy as np

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
    v_channel = hls[:, :, 2]
    v_output = np.zeros_like(v_channel)
    v_output[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_output == 1) & (v_output == 1)] = 1

    return output


def window_mask(w, h, img_ref, center, level):
    out = np.zeros_like(img_ref)
    out[int(img_ref.shape[0] - (level + 1) * h):int(img_ref.shape[0] - level * h),
    max(0, int(center - w)):min(int(center + w), img_ref.shape[1])] = 1
    return out

    # TODO: Insert color threasholding based on HSV here