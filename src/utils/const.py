import os

# region Paths
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# region Dataset
DATASET_PATH = os.path.join(PROJECT_PATH, 'dataset')
# region Raw
DATASET_RAW_PATH = os.path.join(DATASET_PATH, 'raw')
DATASET_RAW_DESCRIPTION_PATH = os.path.join(DATASET_RAW_PATH, 'forms.txt')
DATASET_RAW_IMAGES_PATH = os.path.join(DATASET_RAW_PATH, 'img')
# endregion Raw
# region Prepared
DATASET_PREPARED_PATH = os.path.join(DATASET_PATH, 'prepared')
DATASET_PREPARED_IMAGES_PATH = os.path.join(DATASET_PREPARED_PATH, 'img')
# endregion Prepared
# endregion Dataset
# region Sources
SRC_PATH = os.path.join(PROJECT_PATH, 'src')
# endregion Sources
# region Classifier weights
CLASSIFIER_WEIGHTS_PATH = os.path.join(SRC_PATH, 'weights')
# endregion Classifier weights
# endregion Paths

PCA_RETRY_COUNT = 5
