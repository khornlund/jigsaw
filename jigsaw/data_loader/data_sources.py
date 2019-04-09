import abc
import os
import zipfile
import glob

import kaggle

from jigsaw.data_loader.const import TRAIN_CSV, TEST_CSV, FAST_VEC, GLOVE_TXT


class KaggleDataset(abc.ABC):

    EXPECTED = []

    def __init__(self, data_dir):
        super().__init__()
        self._data_dir = data_dir

    def load(self):
        if self.missing_files():
            kaggle.api.authenticate()
            self.get_data()
        missing_files = self.missing_files()
        if missing_files:
            raise Exception(f'Could not populate {missing_files}!')
        return self.read_data()

    @abc.abstractmethod
    def get_data(self):
        pass

    @abc.abstractmethod
    def read_data(self):
        pass

    def missing_files(self):
        for e in self.EXPECTED:
            f = os.path.join(self._data_dir, e)
            if not os.path.exists(f):
                yield f

    def unzip_all(self, path):
        for zip_f in glob.glob(os.path.join(path, '*.zip')):
            print(f'unzipping: {zip_f}')
            with zipfile.ZipFile(zip_f) as fh:
                for member in fh.namelist():
                    filename = os.path.basename(member)
                    # skip directories
                    if not filename:
                        continue

                    # copy file (taken from zipfile's extract)
                    source = fh.open(member)
                    target = open(os.path.join(path, filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)


class JigsawDataset(KaggleDataset):

    COMPETITION = 'jigsaw-unintended-bias-in-toxicity-classification'

    EXPECTED = [
        TRAIN_CSV,
        TEST_CSV
    ]

    def get_data(self):
        dest = os.path.join(self._data_dir, self.COMPETITION)
        kaggle.api.competition_download_files(self.COMPETITION, path=dest, quiet=False)
        self.unzip_all(dest)

    def read_data(self):
        return 0


class GloveDataset(KaggleDataset):

    DATASET = 'takuok/glove840b300dtxt'

    EXPECTED = [GLOVE_TXT]

    def get_data(self):
        kaggle.api.dataset_download_files(
            self.DATASET, path=self._data_dir, unzip=True, quiet=False)

    def read_data(self):
        return 0

class FastTextDataset(KaggleDataset):

    DATASET = 'yekenot/fasttext-crawl-300d-2m'

    EXPECTED = [FAST_VEC]

    def get_data(self):
        kaggle.api.dataset_download_files(
            self.DATASET, path=self._data_dir, unzip=True, quiet=False)

    def read_data(self):
        return 0
    