from torchvision import datasets, transforms
from torch.utils import data

from jigsaw.base.base_data_loader import BaseDataLoader
from jigsaw.data_loader.data_sources import DataSources
from jigsaw.data_loader.data_processor import PreProcessor


class JigsawDataLoader(BaseDataLoader):
    """"""

    def __init__(self, data_dir, batch_size, shuffle, validation_split=0.0, num_workers=0,
                 max_len=220, training=True):
        self._data_dir = data_dir
        self._max_len = max_len

        # check datasets exist, download if they don't
        DataSources(self._data_dir)

        # get preprocessed data
        (x_train_torch,
        x_test_torch,
        y_train_torch,
        self.y_aux_train,
        self.embedding_matrix) = PreProcessor(self._data_dir, self._max_len).get()

        self.output_dim = y_train_torch.shape[-1]

        self.dataset = (data.TensorDataset(x_train_torch, y_train_torch)
                        if training else
                        data.TensorDataset(x_test_torch))

        super(JigsawDataLoader, self).__init__(
            self.dataset,
            batch_size,
            (shuffle and training),  # don't shuffle for testing
            validation_split,
            num_workers)







