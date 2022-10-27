import numpy as np
from glob import glob
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def get_all_train_path():
    for name in os.walk("./datasets/CASIA/"):
        print(name)

if __name__ == "__main__":
    get_all_train_path()