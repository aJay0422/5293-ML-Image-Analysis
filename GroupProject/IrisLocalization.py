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

    # Step 1: Estimate an approximate center
    proj_h = np.mean(img, axis=0)   # horizontal projection
    proj_v = np.mean(img, axis=1)   # vertical projection
    X_p = np.argmin(proj_h)
    Y_p = np.argmin(proj_v)


    # Step 2: Binarize

    ## find the rectangle area
    coord_upperleft = (X_p - 60, Y_p - 60)
    coord_bottomright = (X_p + 60, Y_p + 60)   # the rectangle area has size (121, 121)

    ## binarize with some threshold
    threshold = 50
    pupil_coords = []   # record all coords with gray scale smaller than threshold
    for x in range(coord_upperleft[0], coord_bottomright[0] + 1):
        for y in range(coord_upperleft[1], coord_bottomright[1] + 1):
            if img[y, x] <= threshold:
                pupil_coords.append([x, y])
    pupil_coords = np.array(pupil_coords)
    centroid = np.mean(pupil_coords, axis=0)   # find the centroid


if __name__ == "__main__":
    file_path = "datasets/CASIA/001/1/001_1_1.bmp"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_smooth = gaussian_smooth(img, kernel_size=7, sigma=1)
    cv2.imshow("Image", img)
    cv2.imshow("Smooth", img_smooth)
    cv2.waitKey(0)

