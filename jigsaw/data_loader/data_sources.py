import abc

import kaggle


class KaggleDataset(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def get(cls):
        pass

class JigsawDataset(KaggleDataset):

    COMPETITION = 'jigsaw-unintended-bias-in-toxicity-classification'

    @classmethod
    def get(cls, path):
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(cls.COMPETITION, path=path, quiet=False)


class GloveDataset(KaggleDataset):

    DATASET = 'takuok/glove840b300dtxt'

    @classmethod
    def get(cls, path):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(cls.DATASET, path=path, unzip=True, quiet=False)


class FastTextDataset(KaggleDataset):

    DATASET = 'yekenot/fasttext-crawl-300d-2m'

    @classmethod
    def get(cls, path):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(cls.DATASET, path=path, unzip=True, quiet=False)