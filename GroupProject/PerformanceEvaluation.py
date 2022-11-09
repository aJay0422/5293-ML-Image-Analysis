import numpy as np
import sklearn.metrics
from joblib import load
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import norm
from scipy.spatial import distance as dst
from tqdm import tqdm


def Identification(metric="l2", n_comp=107, original=False):
    # Prepare lda and classifier
    lda = load("./clf_lda/lda.joblib")
    if original:
        clf = load(f"./clf_lda/clf_original_{metric}.joblib")
    else:
        clf = load(f"./clf_lda/clf_{metric}_{n_comp}.joblib")

    if metric == "l1":
        metric = "cityblock"
    elif metric == "l2":
        metric = "euclidean"

    # Prepare data
    test_features_lda = {}
    for degree in [-9, -6, -3, 0, 3, 6, 9]:
        X_test = np.load(f"test_features_{degree}.npy")
        X_test_lda = lda.transform(X_test)
        if original:
            X_test_lda_n = X_test
        else:
            X_test_lda_n = X_test_lda[:, :n_comp]
        test_features_lda[degree] = X_test_lda_n
    Y_test = np.repeat(np.arange(108), 4)

    # Prediction and evaluation
    dist_matrices = []
    for degree in [-9, -6, -3, 0, 3, 6, 9]:
        dist_m = dst.cdist(test_features_lda[degree], clf.centroids_, metric=metric)   # (432, 108)
        dist_matrices.append(dist_m)
    dist_m = np.min(dist_matrices, axis=0)   # (432, 108)
    Y_pred = np.argmin(dist_m, axis=1)
    crr = np.mean(Y_pred == Y_test)
    # print(sklearn.metrics.classification_report(Y_test, Y_pred))
    return crr


def Verification(threshold=0.5):
    clf = load("./clf_lda/clf_cosine_107.joblib")
    lda = load("./clf_lda/lda.joblib")

    # Prepare data from 7 different rotation degree
    test_features_lda = {}
    for degree in [-9, -6, -3, 0, 3, 6, 9]:
        X_test = np.load(f"test_features_{degree}.npy")
        X_test_lda = lda.transform(X_test)
        test_features_lda[degree] = X_test_lda
    label_test = np.arange(108)
    label_test = np.repeat(label_test, 4)

    # Matching
    centroids = clf.centroids_
    dist_matrices = []
    # calculate distance matrix for 7 different rotation
    for degree in [-9, -6, -3, 0, 3, 6, 9]:
        X_test_lda = test_features_lda[degree]
        dist_m = dst.cdist(X_test_lda, centroids, metric="cosine")
        dist_matrices.append(dist_m)
    dist_matrices = np.array(dist_matrices)
    dist_m = np.min(dist_matrices, axis=0)

    match = 0
    false_match = 0
    nonmatch = 0
    false_nonmatch = 0

    # record false match and false non-match cases
    for i in range(432):
        x_label = label_test[i]
        for cls in range(108):
            distance = dist_m[i, cls]
            if x_label != cls:
                nonmatch += 1
                if distance < threshold:
                    false_match += 1
            else:
                match += 1
                if distance >= threshold:
                    false_nonmatch += 1


    fmr = false_match / nonmatch
    fnmr = false_nonmatch / match
    return fmr, fnmr


def plot_curve(name="crr_dim"):
    if name == "crr_dim":
        metrics = ["l1", "l2", "cosine"]
        n_comps = np.concatenate([np.arange(20, 107, 10), np.array([107])])
        crrs = {}
        crrs["l1"] = []
        crrs["l2"] = []
        crrs["cosine"] = []

        for m in metrics:
            for n_comp in n_comps:
                crr = Identification(m, n_comp)
                crrs[m].append(crr)

        plt.plot(n_comps, crrs["l1"], ".-", label="l1")
        plt.plot(n_comps, crrs["l2"], ".-", label="l2")
        plt.plot(n_comps, crrs["cosine"], ".-", label="cosine")
        plt.xlabel("Dimensionality of the feature vector")
        plt.ylabel("Correct recognition rate")
        plt.legend()
        plt.show()
    elif name == "roc":
        thresholds = np.log10(np.linspace(1, 10, 100))
        x = []
        y = []
        for threshold in tqdm(thresholds):
            fmr, fnmr = Verification(threshold)
            x.append(fmr * 100)
            y.append(fnmr * 100)
        fig, ax = plt.subplots()
        plt.plot(x, y)
        ax.set_xscale("log", base=10)
        ax.set_ylim(0, 60)
        plt.xlabel("False match rate (%)")
        plt.ylabel("False nonmatch rate (%)")
        plt.title("ROC Curve")
        plt.show()


def plot_chart(name="identification"):
    if name == "identification":
        crr_original = []
        crr_reduced = []
        for metric in ['l1', 'l2', 'cosine']:
            crr = Identification(metric=metric, original=True)
            crr_original.append(crr)
            crr = Identification(metric=metric, n_comp=107)
            crr_reduced.append(crr)

        df = pd.DataFrame({
            "original": crr_original,
            "reduced": crr_reduced
        })
        df.index = ['l1', 'l2', 'cosine']
        print("Recognition Results using Different Similarity Measures")
        print(df)
    elif name == "thresholds":
        thresholds = [0.559, 0.588, 0.700]
        fmrs = []
        fnmrs = []
        for threshold in thresholds:
            fmr, fnmr = Verification(threshold)
            fmrs.append(fmr*100)
            fnmrs.append(fnmr*100)
        df = pd.DataFrame({
            "Threshold": thresholds,
            "False match rate (%)": fmrs,
            "False non-match rate (%)": fnmrs
        })
        print("False Match and False Nonmatch Rates with Different Threshold Values")
        print(df)





if __name__ == "__main__":
    plot_curve("roc")