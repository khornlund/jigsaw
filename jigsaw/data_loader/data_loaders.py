import multiprocessing

from torchvision import datasets, transforms

from jigsaw.base.base_loader import BaseDataLoader
from jigsaw.data_loader.data_sources import JigsawDataset, GloveDataset, FastTextDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers)


class JigsawDataLoader(BaseDataLoader):
    """"""

    def __init__(self, data_dir):

        # check datasets exist
        JigsawDataSources.get(data_dir)


class TextDataLoader: 

    def get(self, data_dir, filename):
        pass

    def _preprocess(self, data):
        """Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution"""
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

        def clean_special_chars(text, punct):
            for p in punct:
                text = text.replace(p, ' ')
            return text

        data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
        return data


class JigsawDataSources:

    def __init__(self, path):
        datasets = [JigsawDataset(path), GloveDataset(path), FastTextDataset(path)]
        pool = multiprocessing.Pool(processes = len(datasets))
        pool.map(self.get_data, datasets)

    def get_data(self, dataset):
        return dataset.load()