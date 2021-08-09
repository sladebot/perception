# Perception

#### Section: Single Value Decomposition

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



