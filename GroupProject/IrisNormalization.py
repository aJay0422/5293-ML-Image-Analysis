import numpy as np
import math
import cv2

from IrisLocalization import localization


def normalization(img, X_p, Y_p, Rp, X_i, Y_i, Ri):
    """
    Map the iris from Cartesian coordinate to polar coordinates
    :param img: The input image
    :param X_p: x coordinate of pupil center
    :param Y_p: y coordinate of pupil center
    :param Rp: radius of pupil
    :param X_i: x coordinate of iris center
    :param Y_i: y coordinate of iris center
    :param Ri: radius of iris
    """
    M, N = 64, 512
    output = np.zeros((M, N))
    for X in range(N):
        for Y in range(M):
            theta = 2 * np.pi * X / N
            x_p_theta = X_p + Rp * np.cos(theta)
            x_i_theta = X_i + Ri * np.cos(theta)
            y_p_theta = Y_p + Rp * np.sin(theta)
            y_i_theta = Y_i + Ri * np.sin(theta)
            x = x_p_theta + (x_i_theta - x_p_theta) * Y / M
            y = y_p_theta + (y_i_theta - y_p_theta) * Y / M
            x = round(x)
            y = round(y)
            output[Y, X] = img[y, x]
    output = output.astype(np.uint8)
    return output


if __name__ == "__main__":
    file_path = "datasets/CASIA/001/1/001_1_1.bmp"
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    X_p, Y_p, Rp, X_i, Y_i, Ri = localization(img)
    output = normalization(img, X_p, Y_p, Rp, X_i, Y_i, Ri)

    cv2.imshow("image", output)
    cv2.waitKey(0)
