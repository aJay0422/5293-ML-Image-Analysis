def irisMatching(metric="l2", n_comp=107):
    X_train = np.load("train_features.npy")

    label_train = np.arange(108)
    label_train = np.repeat(label_train, 3)
    # train model with orginal data
    metrics = ["l1", "l2", "cosine"]
    n_comp = np.arange(20, 107, 10)
    for m in metrics:
        clf = NearestCentroid(metric=m)
        # original data
        clf.fit(X_train, label_train)
        clf_name = f"clf_original_{m}.joblib"
        dump(clf, clf_name)
        # reduced data(# of features = 107)
        X_train_lda, lda = dim_reduction(X_train, label_train, 107)
        clf.fit(X_train_lda, label_train)

        clf_name = f"clf_{m}_{107}.joblib"
        dump(clf, clf_name)
        lda_name = f"lda_{107}.joblib"
        dump(lda, lda_name)
        # train model with reduced data
        for n in n_comp:
            X_train_lda, lda = dim_reduction(X_train, label_train, n_comp)
            clf = NearestCentroid(metric=m)
            clf.fit(X_train_lda, label_train)

            clf_name = f"clf_{m}_{n}.joblib"
            dump(clf, clf_name)
            lda_name = f"lda_{n}.joblib"
            dump(lda, lda_name)

