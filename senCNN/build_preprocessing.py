import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self, args):
        super(Preprocessing, self).__init__()
        self.data_path = args.data_path
        self.train_path = args.file_path + '/ratings_train.txt'
        self.test_path = args.file_path + '/ratings_test.txt'

    def makeProcessing(self):
        data = pd.read_csv(self.train_path, sep='\t').loc[:, ['document', 'label']]
        data = data.loc[data['document'].isna().apply(lambda elm: not elm), :]
        train, eval = train_test_split(data, test_size=0.2, shuffle=True, random_state=777)
        # train 데이터 tab으로 저장
        train.to_csv(self.data_path + '/train.txt', sep='\t', index=False)
        # valid 데이터 tab으로 저장
        eval.to_csv(self.data_path + '/val.txt', sep='\t', index=False)

        # test data load (document, label 컬럼을 tab으로 구별)
        data = pd.read_csv(self.test_path, sep='\t').loc[:, ['document', 'label']]
        # test data에서 Nan 값 지우고
        data = data.loc[data['document'].isna().apply(lambda elm: not elm), :]
        # test 데이터 tab으로 저장
        data.to_csv(self.data_path + '/test.txt', sep='\t', index=False)

#
# train_data = Path.cwd() / '..' / 'nsmc-master' / 'ratings_train.txt'
#
# # train data load (document, label 컬럼을 tab으로 구별)
# data = pd.read_csv(train_data, sep='\t').loc[:, ['document', 'label']]
# # train data에서 Nan 값 지우고
# data = data.loc[data['document'].isna().apply(lambda elm: not elm), :]
# # train, valid 파일을 8:2로 나눔
# train, eval = train_test_split(data, test_size=0.2, shuffle=True, random_state=777)
# # train 데이터 tab으로 저장
# train.to_csv(Path.cwd() / '..' / 'data_in' / 'train.txt', sep='\t', index=False, header=False)
# # valid 데이터 tab으로 저장
# eval.to_csv(Path.cwd() / '..' / 'data_in' / 'val.txt', sep='\t', index=False, header=False)
#
# # test 데이터 path
# test_data = Path.cwd() / '..' / 'nsmc-master' / 'ratings_test.txt'
# # test data load (document, label 컬럼을 tab으로 구별)
# data = pd.read_csv(test_data, sep='\t').loc[:, ['document', 'label']]
# # test data에서 Nan 값 지우고
# data = data.loc[data['document'].isna().apply(lambda elm: not elm), :]
# # test 데이터 tab으로 저장
# data.to_csv(Path.cwd() / '..' / 'data_in' / 'test.txt', sep='\t', index=False, header=False)
