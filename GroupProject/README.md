# Peer evaluation form

|                     | IrisLocalization | IrisNormalization | ImageEnhancement | FeatureExtraction | IrisMatching | PerformanceEvaluation | IrisRecognition |
|---------------------|------------------|-------------------|------------------|-------------------|--------------|-----------------------|-----------------|
| Shuo Wang(sw3509)   | Yes              |                   |                  | Yes               | Yes          | Yes                   | Yes             |
| Shangjie Du(sd3497) | Yes              | Yes               | Yes              | Yes               |              | Yes                   | Yes             |
# IrisLocalization
1. Locate the center of the pupil using argmin of the mean of grayscale value
in two axis.
2. Use adaptive threshold to locate the pupil area and use HoughCircle algorithm
to determine the radius of the pupil. This step is repeated twice to get an accurate
result.
3. Remove the pupil area and use Canny edge detector together with HoughCircle
to locate the iris center and determine the radius of the iris. The image is blurred
three times using a median filter since the iris edge is not clear

# IrisNormalization
Transform the annular area between pupil and iris to a rectangular image. 
$$I_n(X,Y) = I_o(x,y)$$
$$x = x_p(\theta) + (x_i(\theta) - x_p(\theta))\frac{Y}{M}$$
$$y = y_p(\theta) + (y_i(\theta) - y_p(\theta))\frac{X}{N}$$
$$\theta = 2\pi X/N$$
Here $I_n$ is the output image of size $M \times N$, $(x_p(\theta),y_p(\theta))$ and $(x_i(\theta), y_i(\theta))$ are the coordinates of the inner and outer boundary points in the direction in the original image $I_o$.

# ImageEnhancement
1. Estimate the illumination in each $16 \times 16$ blocks and get an $M \times N$ illumination image.
2. Subtract the illumination estimation from the original image.

# FeatureExtraction
1. Use modified Gabor filters with kernel size $9 \times 9$ to filter the enhanced image.
2. Use mean and variance in $8 \times 8$ blocks of the filtered image to extract a feature vector.

# IrisMatching
1. Use lda to reduce the feature dimension from 1536 to 107.
2. Fit a nearest centroid classifier.
3. Save the lda and the classifier for future usage on the test dataset.

# PerformanceEvaluation
1. In identification mode, we achieved 82% accuracy on the test set.
2. In verification mode
