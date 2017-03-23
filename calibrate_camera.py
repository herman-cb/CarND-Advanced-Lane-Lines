# Calibration of camera
import numpy as np
import cv2
import glob
import pickle

class CameraCalibration:
    def __init__(self, nx, ny):
        self.objp = np.zeros((nx*ny, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        self.nx = nx
        self.ny = ny
        self.obj_points = []
        self.img_points = []
        self.imgs = []
        self.mtx = None
        self.dist = None

    def save_to_file(self, fname):
        pickle.dump(self, open(fname, "wb"))

    def draw_chessboard_corners_for_all_images(self):
        if self.mtx is None:
            raise AssertionError("must call calibrate() first.")
        print("number of images = {}".format(len(self.imgs)))
        for i in range(len(self.imgs)):
            img = self.imgs[i][2]
            corners = self.img_points[i]
            cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, True)
            ofname = self.imgs[i][1]
            print("writing to {}".format(ofname))
            cv2.imwrite(ofname, img)

    def calibrate(self, images):

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = \
                cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            if ret:
                self.obj_points.append(self.objp)  # The same object points apply to all images
                self.img_points.append(corners)
                self.imgs.append((fname,
                                  "output_images/"+fname.split("/")[-1].split(".")[0]+"with_corners.jpg",
                                  img))

        # To get the image shape
        img = cv2.imread(images[0])
        img_size = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, rvecs, tvecs = \
            cv2.calibrateCamera(self.obj_points, self.img_points, img_size, None, None)


if __name__ == "__main__":
    camera_calibration = CameraCalibration(9, 6)
    camera_calibration.calibrate(glob.glob("./camera_cal/calibration*.jpg"))
    camera_calibration.draw_chessboard_corners_for_all_images()
    camera_calibration.save_to_file("camera_cal.pkl")

    img = cv2.imread("./camera_cal/calibration1.jpg")
    undistorted_img = cv2.undistort(img, camera_calibration.mtx, camera_calibration.dist)
    cv2.imwrite("./output_images/calibration1_undistorted.jpg", undistorted_img)
