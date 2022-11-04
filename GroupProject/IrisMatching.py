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


def irisMatching():
    if not os.path.exists("./clf_lda"):
        os.mkdir("./clf_lda")

    X_train = np.load("train_features.npy")

    label_train = np.arange(108)
    label_train = np.repeat(label_train,3)
    X_train_lda, lda = dim_reduction(X_train, label_train)
    lda_name = f"./clf_lda/lda.joblib"
    dump(lda, lda_name)
    n_comps = np.concatenate([np.arange(20, 107, 10), np.array([107])])
    metrics = ['l1', 'l2', 'cosine']

    for m in metrics:
        clf = NearestCentroid(metric=m)
        # original data
        clf.fit(X_train, label_train)
        clf_name = f"./clf_lda/clf_original_{m}.joblib"
        dump(clf, clf_name)
        for n_comp in n_comps:
            X_train_lda_n = X_train_lda[:,:n_comp]
            clf = NearestCentroid(metric=m)
            clf.fit(X_train_lda_n, label_train)

            clf_name = f"./clf_lda/clf_{m}_{n_comp}.joblib"
            dump(clf, clf_name)



if __name__ == "__main__":
    irisMatching()