import torch
import torch.nn as nn


class ElmanRnn(nn.Module):
    """
    An Elman RNN build using RNNCell
    """

    def __init__(self, input_size, hidden_size, batch_first=False):
        """
        Args:
        :param input_size: int size of the input vectors
        :param hidden_size: size of the hidden state vectors
        :param batch_first: whether the 0th dimension is batch
        """
        super(ElmanRnn, self).__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.batch_first = batch_first
        self.hidden_size = hidden_size

    def _initialize_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, x_in, initial_hidden=None):
        """
        The forward pass of the Elman RNN
        :param x_in: an input data tensor. If batch_first shape should be
        (batch_size, seq_size, feature_size) else
        (seq_size, batch_size, feature_size)
        :param initial_hidden: the initial hidden state for the RNN
        :return: hidden tensor at each step. if batch first shape should be
        (batch_size, seq_size, hidden_size) else
        (seq_size, batch_size, hidden_size)
        """

        if self.batch_first:
            batch_size, seq_size, feature_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feature_size = x_in.size()

        hidden = []
        if initial_hidden is None:
            initial_hidden = self._initialize_hidden(batch_size)
            initial_hidden = initial_hidden.to(x_in.device)

        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hidden.append(hidden_t)
        hidden = torch.stack(hidden)

        if self.batch_first:
            hidden = hidden.permute(1, 0, 2)
        return hidden


class SurnameClassifier(nn.Module):
    """
    A classifier with an RNN to extract features and an MLP to classify
    """

    def __init__(self, embedding_size, vocab_size, num_classes, rnn_hidden_size, batch_first=True, padding_idx=0):
        """

        :param embedding_size: The size of the character embeddings
        :param vocab_size: The number of character in vocab
        :param num_classes: the size of the prediction vector
        :param rnn_hidden_size: The size of the hidden state
        :param batch_first: Informs whether the input tensors will have batch or sequence on the 0th dimension
        :param padding_idx: int The index for the tensor padding
        """
        super(SurnameClassifier, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=padding_idx)
        self.rnn = ElmanRnn(input_size=embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)
        self.fc1 = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(rnn_hidden_size, num_classes)

    def column_gather(self, y_out, x_lengths):
        """
        Get a specifict vector from each batch datapoint in y_out
        :param y_out: shape (batch_size, seq_length, feature)
        :param x_length: shape (batch_size,)
        :return:  shape (batch, feature)
        """
        x_lengths = x_lengths.long().detach().cpu().numpy() - 1
        out = []
        for batch_index, column_index in enumerate(x_lengths):
            out.append(y_out[batch_index][column_index])
        return torch.stack(out)

    def forward(self, x_in, x_lengths=None, apply_softmax=False):
        """
        The forward pass of the classifier
        :param x_in: an input data tensor. shape should be (batch, input_dim)
        :param x_lengths: tensor, the lengths of each sequence in the batch. Using to find the final vector of each sequence.
        :param apply_softmax: bool, a flag for the softmax activation, should be false if used with the cross-entropy loss
        :return: the resulting tensor. shape should be (batch_size, num_classes)
        """
        x_embedded = self.emb(x_in)
        y_out = self.rnn(x_embedded)
        if x_lengths is not None:
            y_out = self.column_gather(y_out, x_lengths)
        else:
            y_out = y_out[:, -1, :]

        y_out = torch.relu(self.fc1(self.dropout(y_out)))
        y_out = self.fc2(self.dropout(y_out))

        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)

        return y_out


# print(
#     SurnameClassifier(embedding_size=10, vocab_size=20, num_classes=10, rnn_hidden_size=23, batch_first=True, padding_idx=0)(
#         torch.randint(high=19, low=1, size=(64, 17))
#     ).shape
# )
