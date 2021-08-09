# Perception

#### Section: Single Value Decomposition

We will try a Single Value Decomposition on an image with variable
ranks to see how much information is being stored at each rank for image compression.

As we know, ```X = UΣV^T```, and we want to figure out the SVD for the image i.e. X is the grayscale image

Original image ( converted to grayscale )
Size: 476KB
!["Original"](images/original.jpg)

Rank: 5
Size: 129KB
!["Rank 5"](images/svd-rank5.jpg)

Rank: 20
Size: 170KB
!["Rank 20"](images/svd-rank20.jpg)

Rank: 100
Size: 187KB
!["Rank 100"](images/svd-rank100.jpg)

Single Values
!["Single Values"](images/rank-plot.jpg)

Single Values: Cumulative
!["Single Values"](images/rank-plot-cumulative.jpg)


