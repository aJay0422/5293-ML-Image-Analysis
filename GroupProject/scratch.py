import cv2
import numpy as np

# load image
file_path = "datasets/CASIA/014/1/014_1_3.bmp"
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(file_path)

# find an approximate center of pupil
proj_h = np.mean(img, axis=0)
proj_v = np.mean(img, axis=1)
X_p = np.argmin(proj_h)
Y_p = np.argmin(proj_v)

img_color = cv2.circle(img_color, (X_p, Y_p), radius=1, color=(0,255,0), thickness=2)
cv2.imshow("Center", img_color)
cv2.waitKey(0)

# crop a 60x60 area around the center
coord_upperleft = (X_p - 60, Y_p - 60)
coord_bottomright = (X_p + 60, Y_p + 60)

## binarize with some threshold
threshold = 50
pupil_coords = []
for x in range(coord_upperleft[0], coord_bottomright[0] + 1):
    for y in range(coord_upperleft[1], coord_bottomright[1] + 1):
        if img[y, x] <= threshold:
            pupil_coords.append([x, y])

# find the centroid of pixels whose gray scale is below the threshold
centroid = np.mean(np.array(pupil_coords), axis=0, dtype=np.int64)
X_p, Y_p = centroid

# repeat binarize again and find a better estimate of pupil center
coord_upperleft = (X_p - 60, Y_p - 60)
coord_bottomright = (X_p + 60, Y_p + 60)
threshold = 50
pupil_coords = []
for x in range(coord_upperleft[0], coord_bottomright[0] + 1):
    for y in range(coord_upperleft[1], coord_bottomright[1] + 1):
        if img[y, x] <= threshold:
            pupil_coords.append([x, y])
centroid = np.mean(np.array(pupil_coords), axis=0, dtype=np.int64)
X_p, Y_p = centroid

# crop a 120x120 area around the pupil center
img_crop = img[Y_p-60:Y_p+60, X_p-60:X_p+60].copy()
img_crop_color = img_color[Y_p-60:Y_p+60, X_p-60:X_p+60].copy()
img_crop_bin = cv2.inRange(img_crop, 0, 50)
cv2.imshow("Edge", img_crop_bin)
cv2.waitKey(0)

pupil_circles = cv2.HoughCircles(img_crop_bin, cv2.HOUGH_GRADIENT, 1, 300,
                                 minRadius=20, maxRadius=60, param2=1)

if pupil_circles is not None:
    pupil_circles = np.round(pupil_circles[0,:]).astype("int")
    for (x, y, r) in pupil_circles:
        cv2.circle(img_crop_color, (x,y), r, color=(0,255,0), thickness=1)
cv2.imshow("Cropped", img_crop_color)
cv2.waitKey(0)

Rp = pupil_circles[0][2]
cv2.circle(img_color, (X_p, Y_p), Rp, color=(0,255,0), thickness=1)
cv2.circle(img_color, (X_p, Y_p), Rp+53, color=(0,0,255), thickness=1)
cv2.imshow("Pupil", img_color)
cv2.waitKey(0)

img_blurred = img.copy()
# img_blurred = cv2.medianBlur(img, 11)
# img_blurred = cv2.medianBlur(img_blurred, 11)
# img_blurred = cv2.medianBlur(img_blurred, 11)

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
img_sharp = cv2.filter2D(img_blurred, ddepth=-1, kernel=kernel)
img_edge = cv2.Canny(img_blurred , threshold1=50, threshold2=100)
cv2.imshow("Edge", img_edge)
cv2.waitKey(0)




img_color = cv2.rectangle(img_color, coord_upperleft, coord_bottomright, (255,0,0),
              2)
img_color = cv2.circle(img_color, (X_p, Y_p), radius=2, color=[0,0,255])
cv2.imshow("Image", img_color)
cv2.waitKey(0)



