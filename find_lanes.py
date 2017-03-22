# Undistort test images
from calibrate_camera import *
from utils import *
from Tracker import *


cc = pickle.load(open("camera_cal.pkl", "rb"))

mtx = cc.mtx
dist = cc.dist

images = glob.glob("./test_images/test*.jpg")

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    preProcessImage = np.zeros_like(img[:, :, 0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(25, 255))
    c_binary = hls_thresh(img, sthresh=(100, 255), vthresh=(50, 255))
    preProcessImage[(gradx == 1) & (grady == 1) | (c_binary == 1)] = 255

    # Perspective transform area
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

    dst = np.float32([[offset, 0],
                      [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]],
                      [offset, img_size[1]]])

    #     print("src = {}".format(src))
    #     print("dst = {}".format(dst))


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preProcessImage, M, img_size, flags=cv2.INTER_LINEAR)

    window_w = 25
    window_h = 80

    tracker = Tracker(window_w, window_h, margin=25, ym=10 / 720., xm=4 / 384., smooth_factor=15)
    xm_per_pix = tracker.xm_per_pixel
    ym_per_pix = tracker.ym_per_pixel
    window_centroids = tracker.find_window_centroids(warped)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    leftx = []
    rightx = []
    for level in range(0, len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_mask = window_mask(window_w, window_h, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_w, window_h, warped, window_centroids[level][1], level)

        l_points[(l_points == 255) | (l_mask == 1)] = 255
        r_points[(r_points == 255) | (r_mask == 1)] = 255

    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0] - (window_h / 2), 0, -window_h)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    # left_fitx = np.poly1d(res_yvals)
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    # right_fitx = np.poly1d(res_yvals)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_w / 2, left_fitx[::-1] + window_w / 2), axis=0),
                                  np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_w / 2, right_fitx[::-1] + window_w / 2), axis=0),
                                   np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    middle_marker = np.array(
        list(zip(np.concatenate((left_fitx + window_w / 2, right_fitx[::-1] - window_w / 2), axis=0),
                 np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    #     print("left_lane.shape = {}".format(right_lane))
    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)

    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [middle_marker], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(img, 1.0, road_warped, 1.0, 0.0)

    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                              np.array(leftx, np.float32) * xm_per_pix, 2)
    curverad = ((1 + \
                 (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * curve_fit_cr[0])
    #     print("curverad = {}".format(curverad))
    cv2.putText(result, 'Radius of curvature {} m'.format(round(curverad, 3)),
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.putText(result, 'Position: {} of center by {} m'.format(side_pos, round(center_diff, 3)),
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    ofname = "./test_images/tracked{}.jpg".format(idx + 1)  # Test index are 1 based
    cv2.imwrite(ofname, result)