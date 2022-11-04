import cv2
import numpy as np
from scipy import ndimage, stats
import matplotlib.pyplot as plt

def localization(img):
    """

    :param img: A gray image
    :return:
    """

    # Estimate an approximate center
    proj_h = np.mean(img, axis=0)   # horizontal projection
    proj_v = np.mean(img, axis=1)   # vertical projection
    # X_p = np.argmin(proj_h)
    X_p = 60 + np.argmin(proj_h[60:-60])
    # Y_p = np.argmin(proj_v)
    Y_p = 60 + np.argmin(proj_v[60:-60])


    # Crop a 60x60 area around the center
    coord_upperleft = (X_p - 60, Y_p - 60)
    coord_bottom_right = (X_p + 60, Y_p + 60)

    # Binarize the cropped image with some threshold
    img_cropped = img[coord_upperleft[1]: coord_bottom_right[1] + 1, coord_upperleft[0]: coord_bottom_right[0] + 1]
    # plt.hist(img_cropped, bins=256)
    # plt.show()
    threshold = np.bincount(img_cropped.reshape(-1)).argmax()
    pupil_coords = []
    for x in range(coord_upperleft[0], coord_bottom_right[0] + 1):
        for y in range(coord_upperleft[1], coord_bottom_right[1] + 1):
            if img[y, x] <= threshold:
                pupil_coords.append([x, y])

    # Find the centroid of those pixels whose gray scale is below the threshold
    centroid = np.mean(np.array(pupil_coords), axis=0, dtype=np.int64)
    X_p, Y_p = centroid

    # Repeat the crop and binarize process to get a better estimate of pupil center
    coord_upperleft = (X_p - 60, Y_p - 60)
    coord_bottom_right = (X_p + 60, Y_p + 60)
    img_cropped = img[coord_upperleft[1]: coord_bottom_right[1] + 1, coord_upperleft[0]: coord_bottom_right[0] + 1]
    threshold = np.bincount(img_cropped.reshape(-1)).argmax()
    pupil_coords = []
    for x in range(coord_upperleft[0], coord_bottom_right[0] + 1):
        for y in range(coord_upperleft[1], coord_bottom_right[1] + 1):
            if img[y, x] <= threshold:
                pupil_coords.append([x, y])

    # Find the centroid of those pixels whose gray scale is below the threshold
    centroid = np.mean(np.array(pupil_coords), axis=0, dtype=np.int64)
    X_p, Y_p = centroid


    # Find edge of the pupil and use hough circle to get the radius of pupil
    img_crop = img[Y_p-60:Y_p+60, X_p-60:X_p+60].copy()
    img_crop_bin = cv2.inRange(img_crop, 0, int(threshold))
    pupil_circles = cv2.HoughCircles(img_crop_bin, cv2.HOUGH_GRADIENT, 1, 300,
                                     minRadius=20, maxRadius=60, param2=1)
    pupil_circles = np.round(pupil_circles[0,:].astype("int"))
    Rp = pupil_circles[0][2]

    img_blurred = img.copy()
    img_blurred = cv2.medianBlur(img_blurred, 11)
    img_blurred = cv2.medianBlur(img_blurred, 11)
    img_blurred = cv2.medianBlur(img_blurred, 11)

    img_edge = cv2.Canny(img_blurred, threshold1=20, threshold2=30)
    img_edge[:, X_p-Rp:X_p+Rp] = 0
    iris_circles = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 1, 300,
                                     minRadius=90, maxRadius=120, param2=1)
    iris_circles = np.round(iris_circles[0,:].astype("int"))
    circle = iris_circles[0]
    X_i, Y_i, Ri = circle[0], circle[1], circle[2]

    center_dist = np.sqrt((X_i - X_p)**2 + (Y_i - Y_p)**2)
    # if center_dist >= Rp/2:
    if True:
        X_i, Y_i, Ri = X_p, Y_p, Rp+53


    return X_p, Y_p, Rp, X_i, Y_i, Ri






if __name__ == "__main__":
    file_path = "datasets/CASIA/055/1/055_1_3.bmp"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(file_path)

    X_p, Y_p, Rp, X_i, Y_i, Ri = localization(img)

    cv2.circle(img_color, (X_p, Y_p), Rp, color=(255, 0, 0), thickness=1)
    cv2.rectangle(img_color, (X_p-45, Y_p-45), (X_p+45, Y_p+45), (255,255,0), thickness=2)
    cv2.imshow("pupil and iris", img_color)
    cv2.waitKey(0)
