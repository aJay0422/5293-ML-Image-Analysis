import cv2
import numpy as np
from scipy import ndimage


def localization(img):
    """

    :param img: A gray image
    :return:
    """

    # Estimate an approximate center
    proj_h = np.mean(img, axis=0)   # horizontal projection
    proj_v = np.mean(img, axis=1)   # vertical projection
    X_p = np.argmin(proj_h)
    Y_p = np.argmin(proj_v)

    # Crop a 60x60 area around the center
    coord_upperleft = (X_p - 60, Y_p - 60)
    coord_botom_right = (X_p + 60, X_p + 60)

    # Binarize the cropped image with some threshold
    threshold = 50
    pupil_coords = []
    for x in range(coord_upperleft[0], coord_botom_right[0] + 1):
        for y in range(coord_upperleft[1], coord_botom_right[1] + 1):
            if img[y, x] <= threshold:
                pupil_coords.append([x, y])

    # Find the centroid of those pixels whose gray scale is below the threshold
    centroid = np.mean(np.array(pupil_coords), axis=0, dtype=np.int64)
    X_p, Y_p = centroid

    # Repeat the crop and binarize process to get a better estimate of pupil center
    coord_upperleft = (X_p - 60, Y_p - 60)
    coord_botom_right = (X_p + 60, X_p + 60)
    threshold = 50
    pupil_coords = []
    for x in range(coord_upperleft[0], coord_botom_right[0] + 1):
        for y in range(coord_upperleft[1], coord_botom_right[1] + 1):
            if img[y, x] <= threshold:
                pupil_coords.append([x, y])

    # Find the centroid of those pixels whose gray scale is below the threshold
    centroid = np.mean(np.array(pupil_coords), axis=0, dtype=np.int64)
    X_p, Y_p = centroid

    # Find edge of the pupil and use hough circle to get the radius of pupil
    img_crop = img[Y_p-60:Y_p+60, X_p-60:X_p+60].copy()
    img_crop_bin = cv2.inRange(img_crop, 0, 50)
    pupil_circles = cv2.HoughCircles(img_crop_bin, cv2.HOUGH_GRADIENT, 1, 300,
                                     minRadius=20, maxRadius=60, param2=1)
    pupil_circles = np.round(pupil_circles[0,:].astype("int"))
    Rp = pupil_circles[0][2]

    X_i, Y_i, Ri = X_p, Y_p, Rp+55

    return X_p, Y_p, Rp, X_i, Y_i, Ri






if __name__ == "__main__":
    pass
