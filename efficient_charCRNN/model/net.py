import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ops import Flatten, Permute
from gluonnlp import Vocab


class EfficientCharCRNN(nn.Module):
    def __init__(self, args, vocab):
        super(EfficientCharCRNN, self).__init__()
        self._dim = args.word_dim
        self._embedding = nn.Embedding(len(vocab), self._dim, vocab.to_indices(vocab.padding_token))
        self._conv = nn.Conv1d(in_channels=self._dim, out_channels=256, kernel_size=5)
        self._conv1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self._maxpool = nn.MaxPool1d(2, stride=2)
        self._dropout = nn.Dropout()
        self._bilstm = nn.LSTM(self._dim, 128, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            m = x.bernoulli(self._word_dropout_ratio)
            x = torch.where(m == 1, torch.tensor(0).to(x.device), x)

        embedding = self._embedding(x).permute(0, 2, 1)
        r = self._conv(embedding)
        r = self._conv1(r)
        r = self._maxpool(r)
        r = F.relu(r)
        r = self._dropout(r)

        r = self._bilstm(r)
        s = self._dropout(r)
        print(s)
        return s
