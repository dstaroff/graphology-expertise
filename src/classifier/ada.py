from utils.globals import (
    AdaBoostClassifier,
    DecisionTreeClassifier,
)
from .classifier import Classifier


class AdaBoost(Classifier):
    name = 'AdaBoost'

    def __init__(self):
        super().__init__()
        self.classifier = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1),
            n_estimators=200,
        )
