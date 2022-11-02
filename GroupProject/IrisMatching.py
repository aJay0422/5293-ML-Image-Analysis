import glob
import os
import numpy as np
import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm
from joblib import dump, load

from IrisLocalization import localization
from IrisNormalization import normalization
from ImageEnhancement import enhancement
from FeatureExtraction import FeatureExtraction



def dim_reduction(x,y):
    lda = LDA()
    lda.fit(x,y)
    x_lda = lda.transform(x)
    return x_lda, lda


def irisMatching(metric="l2"):
    X_train = np.load("train_features.npy")

    label_train = np.arange(108)
    label_train = np.repeat(label_train,3)
    X_train_lda, lda = dim_reduction(X_train, label_train)

    clf = NearestCentroid(metric=metric)
    clf.fit(X_train_lda, label_train)

    clf_name = f"clf_{metric}.joblib"
    dump(clf, clf_name)
    lda_name = f"lda.joblib"
    dump(lda, lda_name)




if __name__ == "__main__":
    irisMatching("l2")
    irisMatching("l1")
    irisMatching("cosine")