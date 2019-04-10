import multiprocessing
import os

from keras.preprocessing import text, sequence
from torchvision import datasets, transforms
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from jigsaw.base.base_data_loader import BaseDataLoader
from jigsaw.data_loader.data_sources import JigsawDataset, GloveDataset, FastTextDataset
from jigsaw.data_loader.const import (TRAIN_CSV, TEST_CSV, GLOVE_TXT, FAST_VEC,
    CN_COMMENT_TEXT, CN_TARGET, CN_AUX)


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


class PreProcessor:

    x_train_pp     = 'x_train.npy'
    x_test_pp      = 'x_test.npy'
    y_train_pp     = 'y_train.npy'
    y_aux_train_pp = 'y_aux_train.npy'
    emb_mx_pp      = 'embedding_matrix.npy'

    def __init__(self, data_dir, max_len, max_feat=None):
        self._data_dir = data_dir
        self._max_len = max_len
        self._max_feat = max_feat

    def get(self, use_saved=True):
        if use_saved:
            try:
                data = self._load_saved()
                return data
            except:
                pass

        try:
            self._process()  # generate from scratch
            data = self._load_saved()
            return data
        except Exception as ex:  # generation must have failed
            raise Exception(f'Could not load data. {ex}')

    def _load_saved(self):
        x_train_f     = os.path.join(self._data_dir, self.x_train_pp)
        x_test_f      = os.path.join(self._data_dir, self.x_test_pp)
        y_train_f     = os.path.join(self._data_dir, self.y_train_pp)
        y_aux_train_f = os.path.join(self._data_dir, self.y_aux_train_pp)
        emb_mx_f      = os.path.join(self._data_dir, self.emb_mx_pp)

        x_train     = np.load(x_train_f)
        x_test      = np.load(x_test_f)
        y_train     = np.load(y_train_f)
        y_aux_train = np.load(y_aux_train_f)
        emb_mx      = np.load(emb_mx_f)

        x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
        x_test_torch  = torch.tensor(x_test, dtype=torch.long).cuda()
        y_train_torch = torch.tensor(
            np.hstack([y_train[:, np.newaxis], y_aux_train]),
            dtype=torch.float32).cuda()

        return (x_train_torch, x_test_torch, y_train_torch, y_aux_train, emb_mx)

    def _process(self):
        print(f'Beginning processing...')
        train_df = pd.read_csv(os.path.join(self._data_dir, TRAIN_CSV))
        test_df  = pd.read_csv(os.path.join(self._data_dir, TEST_CSV))

        print(f'Cleaning...')
        x_train     = self._clean(train_df[CN_COMMENT_TEXT])[:]
        x_test      = self._clean(test_df[CN_COMMENT_TEXT])[:]
        y_train     = np.where(train_df[CN_TARGET] >= 0.5, 1, 0)[:]
        y_aux_train = train_df[CN_AUX + [CN_TARGET]][:]

        del train_df
        del test_df

        print(f'Tokenizing...')
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(list(x_train) + list(x_test))

        x_train = tokenizer.texts_to_sequences(x_train)
        x_test  = tokenizer.texts_to_sequences(x_test)
        x_train = sequence.pad_sequences(x_train, maxlen=self._max_len)
        x_test  = sequence.pad_sequences(x_test, maxlen=self._max_len)

        self._max_feat = self._max_feat or len(tokenizer.word_index) + 1

        print(f'Loading embeddings...')
        crawl_mx = self._create_embedding(tokenizer, FAST_VEC)
        glove_mx = self._create_embedding(tokenizer, GLOVE_TXT)
        emb_mx = np.concatenate([crawl_mx, glove_mx], axis=-1)

        del crawl_mx
        del glove_mx

        x_train_f     = os.path.join(self._data_dir, self.x_train_pp)
        x_test_f      = os.path.join(self._data_dir, self.x_test_pp)
        y_train_f     = os.path.join(self._data_dir, self.y_train_pp)
        y_aux_train_f = os.path.join(self._data_dir, self.y_aux_train_pp)
        emb_mx_f      = os.path.join(self._data_dir, self.emb_mx_pp)

        print(f'Saving...')
        np.save(x_train_f, x_train)
        np.save(x_test_f, x_test)
        np.save(y_train_f, y_train)
        np.save(y_aux_train_f, y_aux_train)
        np.save(emb_mx_f, emb_mx)

        return (x_train, x_test, y_train, y_aux_train, emb_mx)

    def _clean(self, data):
        """Replaces special characters with whitespace.

        https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

        Parameters
        ----------
        data : pandas.Series
            The series of strings to clean.
        """
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

        def clean_special_chars(s, punct):
            # TODO: this could be must faster
            for p in punct:
                s = s.replace(p, ' ')
            return s

        data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
        return data

    def _create_embedding(self, tokenizer, embedding):
        embedding_f = os.path.join(self._data_dir, embedding)
        mx, unknown = self._build_matrix(tokenizer.word_index, embedding_f)
        print(f'N Unknown Words ({embedding}) : {len(unknown)}')
        return mx

    def _build_matrix(self, word_index, path):
        print(f'Building matrix: "{path}"')
        embedding_index = self._load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        unknown_words = []

        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                unknown_words.append(word)
        return embedding_matrix, unknown_words

    def _load_embeddings(self, path):
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        with open(path) as f:
            return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))


class JigsawDataSources:

    def __init__(self, path):
        datasets = [JigsawDataset(path), GloveDataset(path), FastTextDataset(path)]
        pool = multiprocessing.Pool(processes=len(datasets))
        self.results = pool.map(self.get_data, datasets)

    def get_data(self, dataset):
        return dataset.load()
