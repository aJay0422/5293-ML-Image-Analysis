import cv2
import numpy as np
import math
from scipy.signal import convolve2d

from IrisNormalization import normalization
from IrisLocalization import localization
from ImageEnhancement import enhancement

# hyperparameter
region = 48
ksize = 3

sigma1 = 3
gamma1 = 2

sigma2 = 4.5
gamma2 = 3


# modulating function
def m(x, y, f):
    val = np.cos(2 * np.pi * f * math.sqrt(x ** 2 + y ** 2))
    return val


# spatial filter
def gabor(x, y, sigma_x, sigma_y, f):
    gb = (1 / (2 * math.pi * sigma_x * sigma_y)) * np.exp(-0.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y** 2)) * m(x, y, f)
    return gb

#function to calculate spatial filter over 8x8 blocks
def gabor_filter(ksize, f, sigma_x, sigma_y):
    g_filter = np.zeros((ksize, ksize))
    d = ksize // 2
    for i in range(ksize):
        for j in range(ksize):
            dx = d - i
            dy = d - j
            g_filter[i,j] = gabor(dx, dy, sigma_x, sigma_y, f)
    return g_filter


def get_filtered_image(img):
    img_ROI = img[:region, :]

    filter1 = gabor_filter(9, 1 / 1.5, 3, 1.5)
    filter2 = gabor_filter(9, 1 / 1.5, 4.5, 1.5)

    img_filtered1 = convolve2d(img_ROI, filter1, mode="same")
    img_filtered2 = convolve2d(img_ROI, filter2, mode="same")

    return img_filtered1, img_filtered2


def block_feature_extractor(block):
    (n, m) = block.shape
    block = np.abs(block)
    m = np.mean(block)
    v = np.std(block)

    return m, v


def FeatureExtraction(img):
    n_row, n_col = img.shape
    img_filtered1, img_filtered2 = get_filtered_image(img)

    cv2.imshow("Image1", img_filtered1)
    cv2.imshow("Image2", img_filtered2)
    cv2.waitKey(0)

    block_size = 8
    n_blocks_row = n_row // block_size
    n_blocks_col = n_col // block_size

    features = []
    for IMAGE in [img_filtered1, img_filtered2]:
        for i in range(n_blocks_row):
            for j in range(n_blocks_col):
                block = IMAGE[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_features = block_feature_extractor(block)
                features.append(block_features[0])
                features.append(block_features[1])
    features = np.array(features)
    return features




if __name__ == "__main__":
    file_path = "datasets/CASIA/012/1/012_1_1.bmp"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    X_p, Y_p, Rp, X_i, Y_i, Ri = localization(img)
    output = normalization(img, X_p, Y_p, Rp, X_i, Y_i, Ri)
    img_enhanced = enhancement(output)
    features = FeatureExtraction(img_enhanced)
    stop = None
