import cv2
import numpy as np
import glob
import math
import scipy
from scipy.spatial import distance
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# hyperparameter
region = 40
ksize = 3

sigma1 = 3
gamma1 = 2

sigma2 = 4.5
gamma2 = 3

dim = 8


# modulating function
def m(x, y, f):
    val = np.cos(2 * np.pi * f * math.sqrt(x ** 2 + y ** 2))
    return val


# spatial filter
def gabor(x, y, sigma_x, sigma_y, f):
    gb = (1 / (2 * math.pi * sigma_x * sigma_y)) * np.exp(-0.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y** 2)) * m(x, y, f)
    return gb

#function to calculate spatial filter over 8x8 blocks
def spatial_filters(f,sigma_x,sigma_y):
    s_filter=np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            s_filter[i,j]=gabor((-4+j), (-4+i), sigma_x, sigma_y, f)
    return s_filter
