import cv2
import numpy as np
import math
from scipy.signal import convolve2d
import glob
from tqdm import tqdm

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


def get_filtered_image(img, kernel_size=7):
    img_ROI = img[:region, :]

    filter1 = gabor_filter(kernel_size, 1 / 1.5, 3, 1.5)
    filter2 = gabor_filter(kernel_size, 1 / 1.5, 4.5, 1.5)

    img_filtered1 = convolve2d(img_ROI, filter1, mode="same")
    img_filtered2 = convolve2d(img_ROI, filter2, mode="same")

    return img_filtered1, img_filtered2


def block_feature_extractor(block):
    block = np.abs(block)
    m = np.mean(block)
    v = np.std(block)

    return m, v


def FeatureExtraction(img, kernel_size=9):
    # filter the image with modified gabor filter
    img_filtered1, img_filtered2 = get_filtered_image(img, kernel_size)
    n_row, n_col = img_filtered1.shape

    block_size = 8
    n_blocks_row = n_row // block_size
    n_blocks_col = n_col // block_size

    # extract feature by each 8x8 block
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


def save_feature(train=True, dataset_path=None, theta_init=0):
    # read all images
    if train:
        images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob(dataset_path + '*/1/*.bmp'))]
    else:
        images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob(dataset_path + '*/2/*.bmp'))]

    # extract all features
    features = []
    for image in tqdm(images):
        X_p, Y_p, Rp, X_i, Y_i, Ri = localization(image)
        iris_image = normalization(image, X_p, Y_p, Rp, X_i, Y_i, Ri, theta_init)
        enhanced_image = enhancement(iris_image)
        feature = FeatureExtraction(enhanced_image)
        features.append(feature)

    features = np.array(features)
    # save feature
    if train:
        np.save(f"train_features_{theta_init}.npy", features)
    else:
        np.save(f"test_features_{theta_init}.npy", features)

    print("Saved. Shape={}".format(features.shape))




if __name__ == "__main__":
    pass
