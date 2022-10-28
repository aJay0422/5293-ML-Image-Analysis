import numpy as np
from glob import glob
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid


def reduce_dim(x,y):
    lda = LDA()
    lda.fit(x,y)
    x_lda = lda.transform(x)
    return x_lda,lda
def irisMatching_train:
    train_image = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob('./data/*/1/*.bmp'))]
    features_train = []

    for image in train_image:
        X_p, Y_p, Rp, X_i, Y_i, Ri = localization(image)
        iris_image = normalization(image, X_p, Y_p, Rp, X_i, Y_i, Ri)
        enhanced_image = enhancement(iris_image)
        feature = FeatureExtraction(enhanced_image)
        features_train.append(feature)
    label_train = np.arrange(108)
    np.repeat(label_train,3)
    features_train = np.array(features_train)
    x_lda,lda = reduce_dim(features_train,label_train)

    clf = NearestCentroid()
    clf.fit(x_lda,label_train)
    return clf,lda
if __name__ == "__main__":
    get_all_train_path()