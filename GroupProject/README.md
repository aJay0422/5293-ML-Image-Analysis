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
