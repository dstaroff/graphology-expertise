class DatasetDescriptionNotExistingError(FileNotFoundError):
    def __init__(self):
        super(DatasetDescriptionNotExistingError, self).__init__()

    def __str__(self):
        return 'Dataset description is not existing'
