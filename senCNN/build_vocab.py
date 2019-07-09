import itertools
import pickle
import pandas as pd
import gluonnlp as nlp
from model.utils import Vocab
from mecab import MeCab


class Build_Vocab:
    def __init__(self, args):
        super(Build_Vocab, self).__init__()

        self.file_path = args.file_path
        self.data_path = args.data_path
        self.train_path = self.file_path + '/ratings_train.txt'

    def make_vocab(self):
        tr = pd.read_csv(self.train_path, sep='\t').loc[:, ['document', 'label']]
        tokenizer = MeCab()
        tokenized = tr['document'].apply(lambda elm: tokenizer.morphs(str(elm))).tolist()
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(tokenized))

        list_of_tokens = list(map(lambda elm: elm[0], filter(lambda elm: elm[1] >= 10, counter.items())))
        list_of_tokens = sorted(list_of_tokens)
        list_of_tokens.insert(0, '<pad>')
        list_of_tokens.insert(0, '<unk>')

        tmb_vocab = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

        nlp.embedding.list_sources()
        # wiki.ko 데이터를 fasttext로 벡터화 한 임베딩 가져오기
        embedding = nlp.embedding.create('fasttext', source='wiki.ko')
        tmb_vocab.set_embedding(embedding)
        array = tmb_vocab.embedding.idx_to_vec.asnumpy()

        vocab = Vocab(list_of_tokens, padding_token='<pad>', unknown_token='<unk>', bos_token=None, eos_token=None)
        vocab.embedding = array

        with open(self.data_path + '/' + 'vocab.pkl', mode='wb') as io:
            pickle.dump(vocab, io)

#
# # train path
# train_path = Path.cwd() / '..' / 'nsmc-master' / 'ratings_train.txt'
# # train data를 tab으로 구별 document, label 컬럼으로 불러옴
# tr = pd.read_csv(train_path, sep='\t').loc[:, ['document', 'label']]
# # Mecab 정의
# tokenizer = MeCab()
# # document 열의 데이터를 Mecab의 형태소로 나눈 것들을 list로 변환
# tokenized = tr['document'].apply(lambda elm: tokenizer.morphs(str(elm))).tolist()
# # tokenized 에서 각 단어의 count 저장
# counter = nlp.data.count_tokens(itertools.chain.from_iterable(tokenized))
#
# # counter에서 최소 10번 이상 나온것들을 vocab에 저장
# vocab = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)
#
# nlp.embedding.list_sources()
# # wiki.ko 데이터를 fasttext로 벡터화 한 임베딩 가져오기
# embedding = nlp.embedding.create('fasttext', source='wiki.ko')
#
# # 만든 vocab에 벡터 적용
# vocab.set_embedding(embedding)
#
# # vocab.pkl 저장
# with open(Path.cwd() / '..' / 'data_in' / 'vocab_scn.pkl', mode='wb') as io:
#     pickle.dump(vocab, io)
