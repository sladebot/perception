import cv2
import numpy as np


class Detector:
    def __init__(self, query_img):
        self.cam = cv2.VideoCapture(0)
        self.query_img = cv2.imread(query_img)
        self.h, self.w, _ = self.query_img.shape
        self.query_img_bw = cv2.cvtColor(self.query_img, cv2.COLOR_BGR2GRAY)

    def get_frame(self, mirror=False):
        """
        Returns a frame from the webcam
        @param mirror: bool
        """
        _, img = self.cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('Webcam', img)
        return img

    def detect(self, train_img, show_descriptors=False):
        """
        @param train_img: string - Image to compare with the self.query_image
        @param show_descriptors: bool - Enable showing descriptors
        """
        train_img = cv2.imread(train_img)
        train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=12,
                            key_size=20,
                            multi_probe_level=2)

        search_params = dict(checks=50)  # or pass empty dictionary
        orb = cv2.ORB_create()
        queryKeypoints, queryDescriptors = orb.detectAndCompute(self.query_img_bw, None)
        trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.match(queryDescriptors, trainDescriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if show_descriptors:
            cv2.drawKeypoints(self.query_img, queryKeypoints, self.query_img)
            cv2.drawKeypoints(train_img, trainKeypoints, train_img)
            cv2.imshow("Train img", train_img)
            cv2.imshow("Query img", self.query_img)
        final_img = cv2.drawMatches(self.query_img, queryKeypoints, train_img, trainKeypoints, matches[:50]
                                    , None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        while True:
            cv2.imshow("matches", final_img)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()

    def track_on_webcam(self):
        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(self.query_img, None)
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
                pts = np.float32([[0, 0], [0, self.h], [self.w, self.h], [self.w, 0]]).reshape(-1, 1, 2)
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
    d = Detector("images/detect.jpg")
    d.track_on_webcam()
    # d.detect("images/detect_book.jpg")
