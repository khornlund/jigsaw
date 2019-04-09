import abc
import os
import zipfile
import glob

import kaggle

from jigsaw.data_loader.const import TRAIN_CSV, TEST_CSV


class KaggleDataset(abc.ABC):

    EXPECTED = []

    @classmethod
    @abc.abstractmethod
    def get(cls, data_dir):
        pass

    @classmethod
    def validate(cls, data_dir):
        for e in cls.EXPECTED:
            f = os.path.join(data_dir, e)
            if not os.path.exists(f):
                print(f'ERROR: "{f}" not found!')

    @classmethod
    def unzip_all(cls, path):
        for f in glob.glob(os.path.join(path, '*.zip')):
            target = f.replace('.zip', '')
            if os.path.exists(target):
                continue
            with zipfile.ZipFile(f, 'r') as zh:
                print(f'unzipping: {f}')
                zh.extractall(target)


class JigsawDataset(KaggleDataset):

    COMPETITION = 'jigsaw-unintended-bias-in-toxicity-classification'

    EXPECTED = [
        TRAIN_CSV,
        TEST_CSV
    ]

    @classmethod
    def get(cls, data_dir):
        dest = os.path.join(data_dir, cls.COMPETITION)
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(cls.COMPETITION, path=dest, quiet=False)
        cls.unzip_all(dest)
        cls.validate(data_dir)


class GloveDataset(KaggleDataset):

    DATASET = 'takuok/glove840b300dtxt'

    @classmethod
    def get(cls, data_dir):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(cls.DATASET, path=data_dir, unzip=True, quiet=False)


class FastTextDataset(KaggleDataset):

    DATASET = 'yekenot/fasttext-crawl-300d-2m'

    @classmethod
    def get(cls, data_dir):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(cls.DATASET, path=data_dir, unzip=True, quiet=False)