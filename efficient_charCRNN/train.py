import pickle
import argparse
import torch
import torch.nn as nn

from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from model.utils import split_to_jamo
from model.data import Corpus, Tokenizer
from model.net import EfficientCharCRNN
from gluonnlp.data import PadSequence
from tqdm import tqdm
from model.metric import evaluate, acc
#from build_preprocessing import Preprocessing
from build_vocab import Build_Vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--data_type', default='senCNN')
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--print_freq', default=3000, type=int)
    # parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--word_dim', default=8, type=int)
    parser.add_argument('--word_max_len', default=300, type=int)
    parser.add_argument('--global_step', default=1000, type=int)
    parser.add_argument('--data_path', default='../data_in')
    parser.add_argument('--file_path', default='../nsmc-master')
    # parser.add_argument('--build_preprocessing', default=False)
    # parser.add_argument('--build_vocab', default=False)

    args = parser.parse_args()
    # p = Preprocessing(args)
    # p.makeProcessing()

    # v = Build_Vocab(args)
    # v.make_vocab()

    with open(args.data_path + '/' + 'vocab_char.pkl', mode='rb') as io:
        vocab = pickle.load(io)

    padder = PadSequence(length=args.word_max_len, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_to_jamo, pad_fn=padder)

    model = EfficientCharCRNN(args, vocab)

    epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    global_step = args.global_step

    tr_ds = Corpus(args.data_path + '/train.txt', tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = Corpus(args.data_path + '/val.txt', tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdaDelta(params=model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(opt, patience=5)



if __name__ == '__main__':
    main()
