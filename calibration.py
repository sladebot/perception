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
