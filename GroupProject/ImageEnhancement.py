import numpy as np
import cv2

from IrisLocalization import localization
from IrisNormalization import normalization


def enhancement(img):
    """
    Image Enhancement
    :param img: an image of size 64x512
    """
    M, N = img.shape
    m = M // 16   # 4
    n = N // 16   # 32
    illumination = np.zeros((m, n))   # size (4, 32)
    for i in range(m):
        for j in range(n):
            block = img[i*16:(i+1)*16, j*16:(j+1)*16]
            block_mean = np.mean(block)
            illumination[i,j] = block_mean

    # expand the image to original size
    illumination = cv2.resize(illumination, (N, M), interpolation=cv2.INTER_CUBIC)
    illumination = illumination.astype(np.uint8)
    img = img - illumination

    # histogram equalization
    m = M // 32
    n = N // 32
    for i in range(m):
        for j in range(n):
            block = img[i*32:(i+1)*32, j*32:(j+1)*32]
            block_equal = cv2.equalizeHist(block)
            img[i*32:(i+1)*32, j*32:(j+1)*32] = block_equal

    return img


if __name__ == "__main__":
    file_path = "datasets/CASIA/011/1/011_1_3.bmp"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    X_p, Y_p, Rp, X_i, Y_i, Ri = localization(img)
    output = normalization(img, X_p, Y_p, Rp, X_i, Y_i, Ri)
    img_enhanced = enhancement(output)
    cv2.imshow("output", output)
    cv2.imshow("illumination", img_enhanced)
    cv2.waitKey(0)