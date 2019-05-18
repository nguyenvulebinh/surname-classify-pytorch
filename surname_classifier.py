import torch
import torch.nn as nn


class SurnameClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels):
        """
        Args
        :param initial_num_channels: (int) size of the incoming feature vector
        :param num_classes: size of the output prediction vector
        :param num_channels: constant channel size to use throught network
        """
        super(SurnameClassifier, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            nn.ELU()
        )
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x_in, apply_softmax=False):
        """
        The forward pass of the classifier
        :param x_in: an input data tensor. shape should be (batch, initial_num_channels, max_surname_length)
        :param apply_softmax: bool, a flag for the softmax activation, should be false if used with the cross-entropy loss
        :return: the reusulting tensor. shape should be (batch_size, num_classes)
        """
        feature = self.convnet(x_in).squeeze(dim=2)
        prediction_vector = self.fc(feature)
        if apply_softmax:
            prediction_vector = torch.softmax(prediction_vector, dim=1)
        return prediction_vector

#
# print(
#     SurnameClassifier(initial_num_channels=77, num_classes=18, num_channels=256)(
#         torch.rand((64, 77, 17))
#     ).shape
# )
