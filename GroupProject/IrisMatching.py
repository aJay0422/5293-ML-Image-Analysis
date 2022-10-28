import glob
import os
import numpy as np
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm

from IrisLocalization import localization
from IrisNormalization import normalization
from ImageEnhancement import enhancement
from feature_extraction import FeatureExtraction



def reduce_dim(x,y):
    lda = LDA()
    lda.fit(x,y)
    x_lda = lda.transform(x)
    return x_lda,lda


def irisMatching_train(metric="l2"):
    train_image = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob('./datasets/CASIA/*/1/*.bmp'))]
    features_train = []

    for image in tqdm(train_image):
        X_p, Y_p, Rp, X_i, Y_i, Ri = localization(image)
        # print("Localization finished")
        iris_image = normalization(image, X_p, Y_p, Rp, X_i, Y_i, Ri)
        # print("normalization finished")
        enhanced_image = enhancement(iris_image)
        # print("enhancement finished")
        feature = FeatureExtraction(enhanced_image)
        # print("feature extraction finished")
        features_train.append(feature)

    label_train = np.arrange(108)
    np.repeat(label_train,3)
    features_train = np.array(features_train)
    x_lda,lda = reduce_dim(features_train,label_train)

    clf = NearestCentroid(metric=metric)
    clf.fit(x_lda,label_train)
    return clf,lda

if __name__ == "__main__":
    clf1, lda1 = irisMatching_train(metric="l2")
    # clf2, lda2 = irisMatching_train(metric="l1")
    # clf3, lda3 = irisMatching_train(metric="cosine")
    stop = None