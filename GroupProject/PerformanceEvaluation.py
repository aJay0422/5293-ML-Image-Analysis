import numpy as np
from joblib import load


def PerformanceEvaluation(metric="l2"):
    # Prepare lda and classifier
    lda = load("lda.joblib")
    clf = load(f"clf_{metric}.joblib")

    # Prepare data
    X_test = np.load("test_features.npy")
    Y_test = np.repeat(np.arange(108), 4)
    X_test_lda = lda.transform(X_test)

    # Prediction and evaluation
    Y_pred = clf.predict(X_test_lda)
    crr = np.mean(Y_pred == Y_test)
    print(f"Distance Metric: {metric}  CRR: {crr}")


if __name__ == "__main__":
    PerformanceEvaluation("l1")
    PerformanceEvaluation("l2")
    PerformanceEvaluation("cosine")