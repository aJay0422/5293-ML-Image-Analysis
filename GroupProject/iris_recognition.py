import glob
import cv2
import numpy as np
from IrisLocalization import localization
from IrisNormalization import normalization
from ImageEnhancement import enhancement
from feature_extraction import FeatureExtraction
import os

train_path = [file for file in sorted(glob.glob('./datasets/CASIA/*/1/*.bmp'))]

train_image = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob('./datasets/CASIA/*/1/*.bmp'))]
features_train = []
label_train = np.arange(108)
label_train = label_train.repeat(3)
stop = None

for image in train_image:
    X_p, Y_p, Rp, X_i, Y_i, Ri = localization(image)
    iris_image = normalization(image,X_p, Y_p, Rp, X_i, Y_i, Ri)
    enhanced_image = enhancement(iris_image)
    feature = FeatureExtraction(enhanced_image)
    features_train.append(feature)

print("Training data processed.")

test_image = [cv2.imread(file,cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob('./data/*/2/*.bmp'))]
features_test = []

for image in test_image:
    X_p, Y_p, Rp, X_i, Y_i, Ri = localization(image)
    iris_image = normalization(image,X_p, Y_p, Rp, X_i, Y_i, Ri)
    enhanced_image = enhancement(iris_image)
    feature = FeatureExtraction(enhanced_image)
    features_test.append(feature)

print("Testing data processed.")
