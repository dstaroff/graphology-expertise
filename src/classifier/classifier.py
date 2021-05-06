from utils import (
    Constant,
    logger,
)
from utils.globals import (
    os,
    pickle,
)


class Classifier:
    name: str = 'Classifier'

    def __init__(self, verbose=False):
        self.classifier = None
        self.verbose = verbose
        self.weights_file_path = os.path.join(Constant.CLASSIFIER_WEIGHTS_PATH, f'{self.name}_weights.pkl')

    def train(self, X, Y):
        self.classifier.fit(X, Y)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def load_weights(self):
        with open(file=self.weights_file_path, mode='rb') as weights_file:
            self.classifier = pickle.load(weights_file)

        if self.verbose:
            logger.info(f'{self.name} Classifier weights saved to "{self.weights_file_path}"')

    def save_weights(self):
        with open(file=self.weights_file_path, mode='wb') as weights_file:
            pickle.dump(self.classifier, weights_file)

        if self.verbose:
            logger.info(f'{self.name} Classifier weights loaded from "{self.weights_file_path}"')

    def score(self, X, Y):
        Y_pred = self.classifier.predict(X)

        TP, TN, FP, FN = 0, 0, 0, 0

        for i in range(len(Y_pred)):
            if Y[i] == 1:
                if Y_pred[i] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if Y_pred[i] == 0:
                    TN += 1
                else:
                    FP += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)  # How many authors did we correctly label out of all the authors?
        precision = TP / (TP + FP)  # How many of those who we labeled as author are actually author?
        recall = TP / (TP + FN)  # Of all the people who are author, how many of those we correctly predict?
        f1 = 2 * recall * precision / (recall + precision)
        specifity = TN / (TN + FP)  # Of all the people who are author, how many of those did we correctly predict?

        return (TP, TN, FP, FN), accuracy, precision, recall, f1, specifity
