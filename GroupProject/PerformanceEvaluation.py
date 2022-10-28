from IrisMatching import irisMatching_train
import glob
import cv2
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

def PerformanceEvaluation:
    test_image = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob('./data/*/2/*.bmp'))]
    features_test = []

    for image in test_image:
        X_p, Y_p, Rp, X_i, Y_i, Ri = localization(image)
        iris_image = normalization(image, X_p, Y_p, Rp, X_i, Y_i, Ri)
        enhanced_image = enhancement(iris_image)
        feature = FeatureExtraction(enhanced_image)
        features_test.append(feature)

    print("Testing data processed.")
    clf, lda = iris_Matching_train()

    label_test = np.arrange(108)
    np.repeat(label_test,2)
    features_test = np.array(features_test)

    Xtest_lda = lda.transform(features_test)
    label_pred = clf.predict(Xtest_lda)

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

