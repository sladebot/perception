# Perception

### Section 1: Camera Calibration

### Section 1: Homography & Tracking

We use SIFT & ORB as feature detectors to get matching descriptors between a given image that we're matching with another 
image or with the live webcam.

Usage: 

```python
    detector = Detector("<query-image-path>") # images/detect.jpg as an example
```

`detector.track_on_webcam()` is used to access the webcam, it matches the descriptors and finds the homography matrix `H`, 
and applies it with a bounding box. Here's an example:

![Screen Shot 2021-08-10 at 4 25 12 PM](https://user-images.githubusercontent.com/1413154/128947505-0eb47c78-fd39-4ab2-974b-eee89b26cbef.png)


`detector.detect("<target-image-path>") # e.g. - images/detect_book.jpg` uses the target image to match with the query image
with which the detector was initialized. After finding matching descriptors, it draws lines between them. Here's an example:

![Screen Shot 2021-08-10 at 4 24 24 PM](https://user-images.githubusercontent.com/1413154/128947520-15e316bc-e3d7-401c-8245-8e263f2a6f7f.png)


### Section 2: Single Value Decomposition

We will try a Single Value Decomposition on an image with variable
ranks to see how much information is being stored at each rank for image compression.

As we know, ```X = UΣV^T```, and we want to figure out the SVD for the image i.e. X is the grayscale image

Original image ( converted to grayscale )
Size: 476KB
![original](https://user-images.githubusercontent.com/1413154/128650465-3daf04b0-68b6-4fff-b612-65defafecf9b.jpg)


Rank: 5
Size: 129KB
![svd-rank5](https://user-images.githubusercontent.com/1413154/128650471-63f573cf-0b6b-436c-82c9-dbe29cb5fae1.jpg)


Rank: 20
Size: 170KB
![svd-rank20](https://user-images.githubusercontent.com/1413154/128650475-06edfcc8-0b70-4c41-bfeb-88711a0607d3.jpg)


Rank: 100
Size: 187KB
![svd-rank100](https://user-images.githubusercontent.com/1413154/128650478-eeef589c-70af-4488-a441-989b8518f2a5.jpg)


Single Values
![rank-plot](https://user-images.githubusercontent.com/1413154/128650480-3955d744-199d-4906-a4b8-28b5d5062605.jpg)


Single Values: Cumulative
![rank-plot-cumulative](https://user-images.githubusercontent.com/1413154/128650484-ab9e5dbe-ef2d-4228-b538-6f8f1c6a6058.jpg)



