import warnings

from FeatureExtraction import save_feature
from IrisMatching import irisMatching
from PerformanceEvaluation import plot_curve, plot_chart

DATASET_PATH = "./datasets/CASIA/"


def main():
    warnings.filterwarnings("ignore")
    # Extract features
    print("*" * 100)
    print("Extracting features......")
    # save_feature(train=True, dataset_path=DATASET_PATH)
    # save_feature(train=False, dataset_path=DATASET_PATH)

    # Iris Matching
    irisMatching()

    # Performance Evaluation
    ## Identification
    print("*" * 100)
    plot_chart(name="identification")
    print("*" * 100)
    print("crr-dim Curve shown in the new window")
    print("*" * 100)
    plot_curve(name="crr_dim")
    ## Verification
    plot_chart(name="thresholds")
    print("*" * 100)
    print("Plotting ROC Curve......")
    plot_curve(name="roc")
    print("ROC Curve shown in the new window")


if __name__ == "__main__":
    main()