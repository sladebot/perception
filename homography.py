import cv2
import numpy as np


class Detector:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def get_frame(self, mirror=False):
        _, img = self.cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('Webcam', img)
        return img

    def detect(self, query_img, train_img):
        query_img = cv2.imread(query_img)
        train_img = cv2.imread(train_img)
        query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
        trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(queryDescriptors, trainDescriptors)

        matches = sorted(matches, key=lambda x: x.distance)
        cv2.drawKeypoints(query_img, queryKeypoints, query_img)
        cv2.drawKeypoints(train_img, trainKeypoints, train_img)
        cv2.imshow("Train img", train_img)
        cv2.imshow("Query img", query_img)
        final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:80], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        final_img = cv2.resize(final_img, (1000, 650))

        cv2.imshow("matches", final_img)
        cv2.waitKey(10000)

    def track_on_webcam(self, detector_image="images/detect.jpg"):
        img = cv2.imread(detector_image, cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(img, None)
        FLANNINDEXKDTREE = 1
        index_params = dict(algorithm=FLANNINDEXKDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        while True:
            frame = self.get_frame()
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, desc_frame = sift.detectAndCompute(grayframe, None)
            matches = flann.knnMatch(desc_image, desc_frame, k=2)
            good_points = []

            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            if len(good_points) > 40:
                print("Target acquired")
                query_pts = np.float32([kp_image[m.queryIdx]
                                       .pt for m in good_points]).reshape(-1, 1, 2)

                train_pts = np.float32([kp_frame[m.trainIdx]
                                       .pt for m in good_points]).reshape(-1, 1, 2)

                matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
                # matches_mask = mask.ravel().tolist()
                h, w = img.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
                cv2.drawKeypoints(image=frame, keypoints=kp_frame, outImage=frame)
                cv2.imshow("Homography", homography)
            else:
                print("Not enough matching descriptors to detect")
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()


if __name__ == "__main__":
    d = Detector()
    d.detect(query_img="images/detect.jpg", train_img="images/detect_book.jpg")
