import cv2
import numpy as np


file_path = "datasets/CASIA/001/1/001_1_1.bmp"
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(file_path)

proj_h = np.mean(img, axis=0)
proj_v = np.mean(img, axis=1)

X_p = np.argmin(proj_h)
Y_p = np.argmin(proj_v)


coord_upperleft = (X_p - 120, Y_p - 120)
coord_bottomright = (X_p + 120, Y_p + 120)

img_color = cv2.rectangle(img_color, coord_upperleft, coord_bottomright, (255,0,0),
              2)

## binarize with some threshold
threshold = 50
pupil_coords = []
for x in range(coord_upperleft[0], coord_bottomright[0] + 1):
    for y in range(coord_upperleft[1], coord_bottomright[1] + 1):
        if img[y, x] <= threshold:
            pupil_coords.append([x, y])

centroid = np.mean(np.array(pupil_coords), axis=0, dtype=np.int64)
# img_color = cv2.circle(img_color, centroid, 2, (0,0,255), 3)

img_cropped = img[centroid[1] - 120:centroid[1] + 120, centroid[0] - 120:centroid[0] + 120]
cv2.imshow("Image", img_cropped)

img_cropped = cv2.Canny(img_cropped, threshold1=75, threshold2=150)
circles = cv2.HoughCircles(img_cropped, cv2.HOUGH_GRADIENT, 1, 1, param1=150, param2=1)
circles = np.uint16(np.around(circles))
img_cropped_color = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)
for i in circles[0,-1:]:
    img_cropped_color = cv2.circle(img_cropped_color, (i[0], i[1]), i[2],
                         (0,255,0), 2)



cv2.imshow("Image", img_cropped_color)
cv2.waitKey(0)
# cv2.imshow("Image", img_color)
# cv2.waitKey(0)

