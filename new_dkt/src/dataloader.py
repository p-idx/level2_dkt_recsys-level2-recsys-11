import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def _label_encoding(config, train_data, test_data):
    merge = pd.concat([train_data, test_data], axis=0)

    # 입력 순서 재정렬
    config.cate_cols = [col for col in merge.columns if col in config.cate_cols]
    config.cont_cols = [col for col in merge.columns if col in config.cont_cols]

    col2idx = [{v: k for k, v in enumerate(merge[col].unique())} \
        for col in merge.columns if col in config.cate_cols]

    for i, col in tqdm(enumerate(config.cate_cols), desc='cate label encoding', total=len(config.cate_cols)):
        merge[col] = merge[col].map(col2idx[i])

    config.cate2idx = {v: k for k, v in enumerate(config.cate_cols)}
    config.cont2idx = {v: k for k, v in enumerate(config.cont_cols)}
    
    config.cate_offsets = [merge[col].nunique() for col in merge.columns if col in config.cate_cols]
    return merge.iloc[:len(train_data)], merge.iloc[len(train_data):]


class DKTDataset(Dataset):
    def __init__(self, config, X, y):
        self.config = config
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        
    def __getitem__(self, index):
        user_seq_len = len(self.X[index][0])
        cate_cols = [col.values.copy() + 1 for col in self.X[index] if col.name in self.config.cate_cols] 
        # 시계열 패딩 용 + 1. 이제 unknown 은 1 이 됨. 시계열 부족이 0
            
        cont_cols = [col.values.copy() for col in self.X[index] if col.name in self.config.cont_cols] 
        # cont 는 시계열 패딩을 어떻게..? # 마지막 최근 시계열 값으로 앞 시계열 채우겠습니다. 일단은.

        answer = self.y[index][0].values.astype(np.float32)
        
        if user_seq_len >= self.config.seq_len:
            if cate_cols:
                for i, col in enumerate(cate_cols):
                    cate_cols[i] = col[-self.config.seq_len:]
            if cont_cols: # cont feature 가 없을 수도 있음.
                for i, col in enumerate(cont_cols):
                    cont_cols[i] = col[-self.config.seq_len:]
            mask = np.ones(self.config.seq_len, dtype=np.int64) # Byte Tensor 를 씀
            answer = answer[-self.config.seq_len:]
        else:
            if cate_cols:
                for i in range(len(cate_cols)):
                    new_col = np.zeros(self.config.seq_len, np.int64)
                    new_col[-user_seq_len:] = cate_cols[i]
                    cate_cols[i] = new_col
            if cont_cols:
                for i in range(len(cont_cols)):
                    new_col = np.zeros(self.config.seq_len, dtype=np.float32)
                    # new_col[:seq_len] = cont_cols[i][0] # cont 의 시계열이 부족하면 가장 앞 데이터로 채움.
                    new_col[-user_seq_len:] = cont_cols[i]
                    cont_cols[i] = new_col

            mask = np.zeros(self.config.seq_len, dtype=np.int64)
            mask[-user_seq_len:] = 1
            new_answer = np.zeros(self.config.seq_len, dtype=np.float32)
            new_answer[-user_seq_len:] = answer
            answer = new_answer

        cate_cols = np.array(cate_cols).T.astype(np.int64)
        cont_cols = np.array(cont_cols).T.astype(np.float32)

        return cate_cols, cont_cols, mask, answer

    def __len__(self):
        return len(self.y)


def get_data(config, is_train):
    train_data = pd.read_csv('/opt/ml/level2_dkt_recsys-level2-recsys-11/data/eda_train_data.csv')
    test_data = pd.read_csv('/opt/ml/level2_dkt_recsys-level2-recsys-11/data/eda_test_data.csv')

    le_train_data, le_test_data = _label_encoding(config, train_data, test_data)

    if is_train:
        X = le_train_data.drop('Timestamp', axis=1).groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col not in ['userID', 'answerCode']])
        )

        y = le_train_data.drop('Timestamp', axis=1).groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col == 'answerCode'])
        )
    
    else:
        X = le_test_data.drop('Timestamp', axis=1).groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col not in ['userID', 'answerCode']])
        )

        y = le_test_data.drop('Timestamp', axis=1).groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col == 'answerCode'])
        )

    return X, y


def get_loader(config, X, y, shuffle=False):

    dataset = DKTDataset(
        config=config,
        X=X,
        y=y
    )

    loader = DataLoader(
        dataset=dataset,
        num_workers=8,
        shuffle=shuffle,
        batch_size=config.batch_size
    )

    return loader