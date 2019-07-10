import torch
import torch.nn as nn
from model.ops import MultiChannelEmbedding, ConvolutionLayer, MaxOverTimePooling
from model.utils import Vocab


class SenCNN(nn.Module):
    def __init__(self, num_classes: int, vocab: Vocab) -> None:
        super(SenCNN, self).__init__()
        self._embedding = MultiChannelEmbedding(vocab)
        self._convolution = ConvolutionLayer(300, 300)
        self._pooling = MaxOverTimePooling()
        self._dropout = nn.Dropout()
        self._fc = nn.Linear(300, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self._embedding(x)
        fmap = self._convolution(fmap)
        feature = self._pooling(fmap)
        feature = self._dropout(feature)
        score = self._fc(feature)

        return score
