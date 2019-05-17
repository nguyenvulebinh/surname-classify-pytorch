import torch
import torch.nn as nn


class SurnameClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        a simple multi layer perceptron base classifier
        :param input_size: the size of the input feature vector
        :param hidden_size:
        :param output_size: number class output
        """
        super(SurnameClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x_in, apply_softmax=False, train_mode=True):
        hidden_vector = self.fc1(x_in)
        hidden_vector = torch.relu(hidden_vector)
        hidden_vector = torch.dropout(hidden_vector, p=0.5, train=train_mode)

        output_vector = self.fc2(hidden_vector).squeeze()
        if apply_softmax:
            output_vector = torch.softmax(output_vector, dim=1)
        return output_vector
