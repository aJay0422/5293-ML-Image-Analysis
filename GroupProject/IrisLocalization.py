import cv2
import numpy as np
from scipy import ndimage


def gaussian_kernel(kernel_size, sigma=1):
    """
    Get a gaussian kernel
    """
    size = int(kernel_size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    filter = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal

    return filter


def gaussian_smooth(img, kernel_size, sigma=1):
    filter = gaussian_kernel(kernel_size, sigma)
    img_smooth = ndimage.convolve(img, filter)
    return img_smooth


def sobel_kernel():
    pass


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
    for x in range(coord_upperleft[0], coord_botom_right[0] + 1)
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
    for x in range(coord_upperleft[0], coord_botom_right[0] + 1)
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
    pupil_circles = np,round(pupil_circles[0,:].astype("int"))
    Rp = pupil_circles[0][2]






if __name__ == "__main__":
    file_path = "datasets/CASIA/001/1/001_1_1.bmp"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_smooth = gaussian_smooth(img, kernel_size=7, sigma=1)
    cv2.imshow("Image", img)
    cv2.imshow("Smooth", img_smooth)
    cv2.waitKey(0)

