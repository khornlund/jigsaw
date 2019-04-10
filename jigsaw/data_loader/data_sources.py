import abc
import os
import zipfile
import glob
import shutil
import multiprocessing

import kaggle

from jigsaw.data_loader.const import TRAIN_CSV, TEST_CSV, FAST_VEC, GLOVE_TXT


class DataSources:

    def __init__(self, path):
        datasets = [JigsawDataset(path), GloveDataset(path), FastTextDataset(path)]
        pool = multiprocessing.Pool(processes=len(datasets))
        self.results = pool.map(self.get_data, datasets)

    def get_data(self, dataset):
        return dataset.load()


# -- Abstract base classes ----------------------------------------------------

class KaggleDataset(abc.ABC):

    expected_f = []

    def __init__(self, data_dir):
        super().__init__()
        self._data_dir = data_dir

    def load(self):
        print(f'Loading {self.__class__.__name__}')
        if self.missing_files():
            try:
                kaggle.api.authenticate()
                self.get_data()
            except Exception as ex:
                raise Exception(f'Caught exception "{ex}". Ensure you have kaggle API '
                                f'set up properly: "https://github.com/Kaggle/kaggle-api"')
            missing_files = self.missing_files()
            if missing_files:
                raise Exception(f'Could not populate {missing_files}!')
            print(f'Downloaded: {self.expected_f}')
        return self.read_data()

    @abc.abstractmethod
    def get_data(self):
        pass

    @abc.abstractmethod
    def read_data(self):
        pass

    def missing_files(self):
        missing = []
        for e in self.expected_f:
            f = os.path.join(self._data_dir, e)
            if not os.path.exists(f):
                missing.append(f)
        return missing


class KaggleCompetitionDataset(KaggleDataset):

    def get_data(self):
        dest = os.path.join(self._data_dir, self.competition)
        kaggle.api.competition_download_files(self.competition, path=dest, quiet=False)
        self.unzip_all(dest)

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


class KaggleUserDataset(KaggleDataset):

    def get_data(self):
        out_f = os.path.join(self._data_dir, self.dataset)
        kaggle.api.dataset_download_files(
            self.user + '/' + self.dataset,
            path=out_f, unzip=True, quiet=False)


# -- Datasets -----------------------------------------------------------------

class JigsawDataset(KaggleCompetitionDataset):

    competition = 'jigsaw-unintended-bias-in-toxicity-classification'
    expected_f  = [TRAIN_CSV, TEST_CSV]

    def read_data(self):
        return 0


class GloveDataset(KaggleUserDataset):

    user       = 'takuok'
    dataset    = 'glove840b300dtxt'
    expected_f = [GLOVE_TXT]

    def read_data(self):
        return 0


class FastTextDataset(KaggleUserDataset):

    user       = 'yekenot'
    dataset    = 'fasttext-crawl-300d-2m'
    expected_f = [FAST_VEC]

    def read_data(self):
        return 0
