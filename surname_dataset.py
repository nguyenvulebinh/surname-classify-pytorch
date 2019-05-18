import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class Vocabulary(object):
    """
    Class to process text and extract Vocabulary for mapping
    """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        :param token_to_idx:  a pre-existing map of tokens to indices
        :param add_unk: a flag that indicates whether to add the UNK token
        :param unk_token: the UNK token to add into the Vocabulary
        """
        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx

        self._idx_to_token = {token: idx for idx, token in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """
        return a dictionary that can be serialized
        :return:
        """
        return {
            'token_to_idx': self._token_to_idx,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token
        }

    @classmethod
    def from_serializable(cls, contents):
        """
        instantiates the Vocabulary from a serialized dictionary
        :param contents:
        :return:
        """
        return cls(**contents)

    def add_token(self, token):
        """
        Update mapping dicts based on the token
        :param token: str, the item to add into the Vocabulary
        :return:
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """
        Retrieve the index associated with the token or the UNK index if token isn't present
        :param token: (str) the token to look up
        :return: the index corresponding to the token
        Notes: 'unk_index' needs to be >= 0 (having been added into the Vocabulary)
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """
       Return the token associated with the index
       :param index: the index to lookup
       :return: the token corresponding to the index
       Raises KeyError if the index is not in the Vocabulary
       """
        if index not in self._idx_to_token:
            raise KeyError("The index {} is not in the Vocabulary".format(index))
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size={})>".format(len(self))

    def __len__(self):
        return len(self._token_to_idx)


class SurnameVectorizer(object):
    """
    The Vectorizer which coordinates the Vocabulary and puts them to use
    """

    def __init__(self, surname_vocab, nationality_vocab, max_surname_length):
        """

        :param surname_vocab: (Vocabulary) maps surname to integers
        :param nationality_vocab: (Vocabulary) maps class labels to integers
        :param max_surname_length: (int) max length of text surname to create matrix
        """
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        self._max_surname_length = max_surname_length

    def vectorize(self, text):
        """
        Create a collapsed one hot vector for the text
        :param text: surname
        :return: one_hot ndarray the collapsed one-hot encoding
        """
        matrix_vector = np.zeros((len(self.surname_vocab), self._max_surname_length), dtype=np.float32)
        for char_index, char in enumerate(text):
            matrix_vector[self.surname_vocab.lookup_token(char)][char_index] = 1
        return matrix_vector

    @classmethod
    def from_dataframe(cls, dataframe):
        """
        Instantiate the vectorizer from dataset dataframe
        :param dataframe: the surname dataset
        :return: an instance of the SurnameVectorizer
        """
        surname_vocab = Vocabulary(add_unk=True)
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0
        for _, row in dataframe.iterrows():
            max_surname_length = max(len(row.surname), max_surname_length)
            for char in row.surname:
                surname_vocab.add_token(char)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab, max_surname_length)

    @classmethod
    def from_serializable(cls, contents):
        """
        Intantiate a SurnameVectorizer from serializable dictionary
        :param contents: the serializable dictionary
        :return: an instance of the SurnameVectorizer class
        """
        surname_vocab = Vocabulary.from_serializable(contents['surname_vocab'])
        nationality_vocab = Vocabulary.from_serializable(contents['nationality_vocab'])
        max_surname_length = contents['max_surname_length']
        return cls(surname_vocab, nationality_vocab, max_surname_length)

    def to_serializable(self):
        """
        Create the serializable dictionary for caching
        :return: contents the serializable dictionary
        """

        return {
            'surname_vocab': self.surname_vocab.to_serializable(),
            'nationality_vocab': self.nationality_vocab.to_serializable(),
            'max_surname_length': self._max_surname_length
        }


class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        """
        :param surname_df: dataset
        :param vectorizer: vectorizer instantiated from dataset
        """
        self.surname_df = surname_df
        self._vectorizer = vectorizer

        self.train_df = self.surname_df[self.surname_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.surname_df[self.surname_df.split == 'val']
        self.val_size = len(self.val_df)

        self.test_df = self.surname_df[self.surname_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

        # Class weights
        class_counts = surname_df.nationality.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.nationality_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):
        """
        :param surname_csv: csv file path
        :return:
        """
        surname_df = pd.read_csv(surname_csv)
        train_surname_df = surname_df[surname_df.split == 'train']
        return cls(surname_df, SurnameVectorizer.from_dataframe(train_surname_df))

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """
        :param index: (int) the index to the data point
        :return: a dict of the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        surname_matrix = self._vectorizer.vectorize(row.surname)
        nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)
        return {
            'x_data': surname_matrix,
            'y_target': nationality_index
        }

    def get_num_batches(self, batch_size):
        """
        Given a batch size, return the number of batches in the dataset
        :param batch_size:
        :return:
        """
        return len(self) // batch_size


def generate_batches(dataset, batch_size, shuffer=True, drop_last=True, device='cpu'):
    """
    A generator function which wraps the Pytorch DataLoader. It will ensure each tensor is on write device location
    :param dataset:
    :param batch_size:
    :param shuffer:
    :param drop_last:
    :param device:
    :return:
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffer, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
