from torchvision import datasets, transforms

from jigsaw.base import BaseDataLoader
from jigsaw.data_loader import JigsawDataset, GloveDataset, FastTextDataset


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
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class JigsawDataLoader(BaseDataLoader):
    """"""

    def __init__(self, data_dir):

        # check datasets exist
        JigsawDataSources.get(data_dir)


class JigsawDataSources:

    @classmethod
    def get(cls, path):
        JigsawDataset.get(path)
        GloveDataset.get(path)
        FastTextDataset.get(path)
