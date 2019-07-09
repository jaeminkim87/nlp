import torch
import torch.nn as nn
from model.ops import Flatten, Permute
from gluonnlp import Vocab


class CharCNN(nn.Module):
    """CharCNN class"""
    def __init__(self, num_classes: int, embedding_dim: int, vocab: Vocab) -> None:
        """Instantiating CharCNN class

        Args:
            num_classes (int): the number of classes
            embedding_dim (int): the dimension of embedding vector for token
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        super(CharCNN, self).__init__()
        self._embedding = nn.Embedding(len(vocab), embedding_dim, vocab.to_indices(vocab.padding_token))
        self._conv1D_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=7)
        self._conv1D_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7)
        self._conv1D_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self._conv1D_4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self._conv1D_5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self._conv1D_6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self._maxpool1d_1 = nn.MaxPool1d(3, 3)
        self._maxpool1d_2 = nn.MaxPool1d(3, 3)
        self._maxpool1d_3 = nn.MaxPool1d(3, 3)
        self._permute = Permute()

        self._linear_1 = nn.Linear(in_features=1792, out_features=512)
        self._linear_2 = nn.Linear(in_features=512, out_features=512)
        self._linear_3 = nn.Linear(in_features=512, out_features=num_classes)


        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embedding(x)
        #print(x.size())
        x = self._permute(x)
        #print(x.size())
        x = self._conv1D_1(x)
        #print(x.size())
        x = nn.ReLU()(x)
        #print(x.size())
        x = self._maxpool1d_1(x)
        #print(x.size())
        x = self._conv1D_2(x)
        #print(x.size())
        x = nn.ReLU()(x)
        #print(x.size())
        x = self._maxpool1d_2(x)
        #print(x.size())
        x = self._conv1D_3(x)
        #print(x.size())
        x = nn.ReLU()(x)
        #print(x.size())
        x = self._conv1D_4(x)
        #print(x.size())
        x = nn.ReLU()(x)
        #print(x.size())
        x = self._conv1D_5(x)
        #print(x.size())
        x = nn.ReLU()(x)
        #print(x.size())
        x = self._conv1D_6(x)
        #print(x.size())
        x = nn.ReLU()(x)
        #print(x.size())
        x = self._maxpool1d_3(x)
        #print(x.size())
        feature = Flatten()(x)
        #print(feature.size())

        #print(feature)

        linear = self._linear_1(feature)
        linear = nn.ReLU()(linear)
        linear = nn.Dropout()(linear)
        linear = self._linear_2(linear)
        linear = nn.ReLU()(linear)
        linear = nn.Dropout()(linear)
        score = self._linear_3(linear)
        #print(score)

        #feature = self._extractor(x)
        #score = self._classifier(feature)
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)