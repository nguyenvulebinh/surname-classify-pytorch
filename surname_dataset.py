import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json


class Vocabulary(object):
    """
    Class to process text and extract Vocabulary for mapping
    """

    def __init__(self, token_to_idx=None):
        """
        :param token_to_idx:  a pre-existing map of tokens to indices
        """
        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx

        self._idx_to_token = {token: idx for idx, token in self._token_to_idx.items()}

    def to_serializable(self):
        """
        return a dictionary that can be serialized
        :return:
        """
        return {
            'token_to_idx': self._token_to_idx
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


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>", end_seq_token="<END>"):
        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({
            'unk_token': self._unk_token,
            'mask_token': self._mask_token,
            "begin_seq_token": self._begin_seq_token,
            'end_seq_token': self._end_seq_token
        })
        return contents

    def lookup_token(self, token):
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx.get(token)


class SurnameVectorizer(object):
    """
    The Vectorizer which coordinates the Vocabulary and puts them to use
    """

    def __init__(self, surname_vocab, nationality_vocab):
        """

        :param surname_vocab: (Vocabulary) maps surname to integers
        :param nationality_vocab: (Vocabulary) maps class labels to integers
        """
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, text, vector_length=-1):
        """
        Create a collapsed one hot vector for the text
        :param text: surname
        :return: one_hot ndarray the collapsed one-hot encoding
        """
        indices = [self.surname_vocab.begin_seq_index]
        indices.extend([self.surname_vocab.lookup_token(char) for char in text])
        indices.append(self.surname_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.surname_vocab.mask_index

        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, dataframe):
        """
        Instantiate the vectorizer from dataset dataframe
        :param dataframe: the surname dataset
        :return: an instance of the SurnameVectorizer
        """
        surname_vocab = SequenceVocabulary()
        nationality_vocab = Vocabulary()
        max_surname_length = 0
        for _, row in dataframe.iterrows():
            max_surname_length = max(len(row.surname), max_surname_length)
            for char in row.surname:
                surname_vocab.add_token(char)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """
        Intantiate a SurnameVectorizer from serializable dictionary
        :param contents: the serializable dictionary
        :return: an instance of the SurnameVectorizer class
        """
        surname_vocab = SequenceVocabulary.from_serializable(contents['surname_vocab'])
        nationality_vocab = Vocabulary.from_serializable(contents['nationality_vocab'])
        return cls(surname_vocab, nationality_vocab)

    def to_serializable(self):
        """
        Create the serializable dictionary for caching
        :return: contents the serializable dictionary
        """

        return {
            'surname_vocab': self.surname_vocab.to_serializable(),
            'nationality_vocab': self.nationality_vocab.to_serializable(),
        }


class SurnameDataset(Dataset):
    def __init__(self, surname_df, vectorizer):
        """
        :param surname_df: dataset
        :param vectorizer: vectorizer instantiated from dataset
        """
        self.surname_df = surname_df
        self._vectorizer = vectorizer

        self._max_seq_length = max(map(len, self.surname_df.surname)) + 2

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
    def load_dataset_and_load_vectorizer(cls, surname_csv, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            surname_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of SurnameDataset
        """
        surname_df = pd.read_csv(surname_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(surname_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of SurnameVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return SurnameVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

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
        surname_vector, vec_length = self._vectorizer.vectorize(row.surname, self._max_seq_length)
        nationality_index = self._vectorizer.nationality_vocab.lookup_token(row.nationality)
        return {
            'x_data': surname_vector,
            'y_target': nationality_index,
            'x_length': vec_length
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
