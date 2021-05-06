class PCARetryLimitExceededError(RuntimeError):
    def __init__(self):
        super(PCARetryLimitExceededError, self).__init__()

    def __str__(self):
        return 'PCA transformation retry limit exceeded'
