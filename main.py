import argparse

import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src'))

from classifier import (
    Ada,
    KNN,
    SVM,
)
from exception import (
    PCARetryLimitExceededError,
)
from preprocess import (
    get_data_for_image,
    pca_transform,
)
from utils import (
    Constant,
)
from utils.globals import (
    np,
    cv2,
)


def get_args():
    parser = argparse.ArgumentParser(
        description='Graphology expertise tool',
        add_help=True,
    )

    parser.add_argument(
        '-f',
        '--image1',
        required=True,
        action='store',
        help='Path to the first image to expertise',
    )
    parser.add_argument(
        '-s',
        '--image2',
        required=True,
        action='store',
        help='Path to the second image to expertise',
    )
    parser.add_argument(
        '-c',
        '--classifiers',
        required=False,
        nargs='+',
        default=['knn'],
        help='Applied classifiers. Appropriate values: "ada", "knn", "svm"',
    )

    return validate_args(parser.parse_args())


def validate_args(args):
    allowed_classifier_names = ['ada', 'knn', 'svm']

    for classifier in args.classifiers:
        if classifier not in allowed_classifier_names:
            raise ValueError(
                f'Inappropriate classifier name "{classifier}". Classifier must be one of {allowed_classifier_names}')

    args.image1 = os.path.abspath(args.image1)
    args.image2 = os.path.abspath(args.image2)

    if not os.path.exists(args.image1) or not os.path.isfile(args.image1):
        raise FileNotFoundError(f'Image file on path "{args.image1}" does not not exist or is not a file')
    if not os.path.exists(args.image2) or not os.path.isfile(args.image2):
        raise FileNotFoundError(f'Image file on path "{args.image2}" does not not exist or is not a file')

    return args


def fetch_images(image1_path, image2_path) -> (np.array, np.array):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    return image1, image2


def get_lines_for_image(image):
    lines = []
    for line in get_data_for_image(image):
        lines.append(line)

    return np.array(lines)


def get_pca_transformed(X):
    retries = 0
    success = False
    while not success:
        if retries > Constant.PCA_RETRY_COUNT:
            raise PCARetryLimitExceededError()
        try:
            X = pca_transform(X, min_components=6)
            success = True
        except Exception:
            retries += 1

    return X


def get_classifiers(args):
    classifiers = []
    for classifier_name in args.classifiers:
        if classifier_name == 'ada':
            classifier = Ada()
        if classifier_name == 'knn':
            classifier = KNN()
        if classifier_name == 'svm':
            classifier = SVM()

        classifier.load_weights()
        classifiers.append(classifier)

    return classifiers


# noinspection PyUnboundLocalVariable
def main():
    try:
        args = get_args()
    except ValueError as e:
        print(e)
        exit(1)

    image1, image2 = fetch_images(args.image1, args.image2)

    X_1, X_2 = get_lines_for_image(image1), get_data_for_image(image2)
    del image1, image2

    try:
        X_1 = get_pca_transformed(X_1)
    except PCARetryLimitExceededError as e:
        print(e)
        exit(1)

    try:
        X_2 = get_pca_transformed(X_2)
    except PCARetryLimitExceededError as e:
        print(e)
        exit(1)

    X = []
    for i in range(min(len(X_1), len(X_2))):
        X.append(np.concatenate([X_1[i], X_2[i]]))
    X = np.array(X)

    classifiers = get_classifiers(args)
    predictions = []
    for classifier in classifiers:
        Y_pred = classifier.predict_proba(X)

        similar_prediction_score = [preds_per_line[1] for preds_per_line in Y_pred]
        predictions.append(np.median(similar_prediction_score))

    pred = np.median(predictions)

    print(f'These 2 images are written by the same one person with the confidence = {pred:.2%}')


if __name__ == "__main__":
    main()
