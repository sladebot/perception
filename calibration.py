#!/usr/bin/env python

import cv2
import numpy as np
import glob


class Camera:
    def __init__(self, square_size, width=9, height=6, dir_path="images/calibration"):
        self.width = width
        self.height = height
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.images = glob.glob(f"{dir_path}/*.jpeg")
        objp = np.zeros((height*width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        self.objp = objp * square_size

    def calibrate(self, show_board=False):
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        gray_shape = 0
        for fname in self.images:
            print(f"Reading image {fname}")
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(self.objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                imgpoints.append(corners)

                if show_board:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape[::-1], None, None)
        cv2.destroyAllWindows()
        return [ret, mtx, dist, rvecs, tvecs]

    def save_coefficients(self, mtx, dist, path):
        """
        Save camera matrix & distortion coefficients to given path/file
        @param mtx:
        @param dist:
        @param path:
        """
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write('K', mtx)
        cv_file.write('D', dist)
        cv_file.release()

    def load_coefficients(self, path):
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode("K").mat()
        dist_matrix = cv_file.getNode("D").mat()
        cv_file.release()

        return [camera_matrix, dist_matrix]

    def undistort(self, image, config):
        mtx, dist = self.load_coefficients(config)
        original = cv2.imread(image)
        return cv2.undistort(original, mtx, dist, None, None)
        # cv2.imwrite("undistorted.jpg", dst)


if __name__ == "__main__":
    # Parameters
    IMAGES_DIR = "images/calibration/"
    IMAGES_FORMAT = '.jpeg'
    SQUARE_SIZE = 1.6
    WIDTH = 6
    HEIGHT = 9

    camera = Camera(
        SQUARE_SIZE,
        WIDTH,
        HEIGHT
    )
    ret, matrix, distortion, r_vecs, t_vecs = camera.calibrate(show_board=False)
    # Displaying required output
    print(" Camera matrix:")
    print(matrix)

    print("\n Distortion coefficient:")
    print(distortion)

    print("\n Rotation Vectors:")
    print(r_vecs)

    print("\n Translation Vectors:")
    print(t_vecs)

    camera.save_coefficients(matrix, distortion, "configs/calibration.yml")
    dst = camera.undistort(image="images/calibration/D1D741DB-E96A-48F9-A2A8-CBBF32A6E247_1_105_c.jpeg", config="configs/calibration.yml")
    src = cv2.imread("images/calibration/D1D741DB-E96A-48F9-A2A8-CBBF32A6E247_1_105_c.jpeg")
    cv2.imshow("Undistorted", dst)
    cv2.imshow("Distorted", src)
    cv2.waitKey(10000)
















#
# # Defining the dimensions of checkerboard
# CHECKERBOARD = (6, 9)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
# # Creating vector to store vectors of 3D points for each checkerboard image
# objpoints = []
# # Creating vector to store vectors of 2D points for each checkerboard image
# imgpoints = []
#
# # Defining the world coordinates for 3D points
# objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# prev_img_shape = None
#
# # Extracting path of individual image stored in a given directory
# images = glob.glob('./images/*.jpeg')
# for fname in images:
#     print(f"File - {fname}")
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     # If desired number of corners are found in the image then ret = true
#     ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
#                                              cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
#
#     """
#     If desired number of corner are detected,
#     we refine the pixel coordinates and display
#     them on the images of checker board
#     """
#     if ret == True:
#         objpoints.append(objp)
#         # refining pixel coordinates for given 2d points.
#         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#
#         imgpoints.append(corners2)
#
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
#
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
#
# cv2.destroyAllWindows()
# print("Read images & destroyed windows")
# h, w = img.shape[:2]
#
# """
# Performing camera calibration by
# passing the value of known 3D points (objpoints)
# and corresponding pixel coordinates of the
# detected corners (imgpoints)
# """
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
# print("Camera matrix : \n")
# print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)
