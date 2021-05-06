class DatasetNotExistingError(FileNotFoundError):
    def __init__(self):
        super(DatasetNotExistingError, self).__init__()

    def __str__(self):
        return 'Dataset folder is not existing'
