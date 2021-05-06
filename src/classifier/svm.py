from utils.globals import svm
from .classifier import Classifier


class SVM(Classifier):
    name = 'SVM'

    def __init__(self, kernel='linear', C=4000, degree=1):
        super().__init__()
        self.classifier = svm.SVC(
            kernel=kernel,
            C=C,
            gamma='scale',
            probability=True,
            degree=degree,
            tol=0.9999,
        )
