from utils.globals import KNeighborsClassifier
from .classifier import Classifier


class KNN(Classifier):
    name = 'KNN'

    def __init__(self):
        super().__init__()
        self.classifier = KNeighborsClassifier(n_neighbors=12)
