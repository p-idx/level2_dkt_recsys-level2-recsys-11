import os
from datetime import datetime

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple


def load_data(args):
    if args.new:
        train_df = pd.read_csv(os.path.join(args.data_dir, 'le_EDA_new_train_data.csv'))
        num = train_df['userID'].nunique()
        print(f'this train is new!{num}')
    else:
        train_df = pd.read_csv(os.path.join(args.data_dir, 'le_EDA_train_data.csv'))
        num = train_df['userID'].nunique()
        print(f'this train is basic!{num}')
    test_df = pd.read_csv(os.path.join(args.data_dir, 'le_EDA_test_data.csv'))
    
    if args.merge:
        train_df = pd.concat([train_df, test_df[test_df['answerCode'] != -1]])
    
    feature_config = {}

    if args.new:
        f = open(os.path.join(args.data_dir, 'new_feature_config.txt'), 'r')
    else:
        f = open(os.path.join(args.data_dir, 'feature_config.txt'), 'r')

    for line in f:
        num, name = line.split(',')
        feature_config[num] = name.strip()
    f.close()

    if args.fe[0] == 'all':
        cols = list(feature_config.values())
    else:
        cols = [feature_config[num] for num in args.fe]

    train_df = train_df[['userID', 'answerCode'] + cols]
    test_df = test_df[['userID', 'answerCode'] + cols]

    cate_cols = [col_name for col_name in train_df.columns if col_name[-2:] == '_c']
    args.cate_num = len(cate_cols)
    args.cont_num = len(train_df.columns) - args.cate_num - 2 # userID, answerCode

    args.offsets = []
    offset = 0
    for cate_col in cate_cols:
        train_df[cate_col] += offset
        test_df[cate_col] += offset
        offset = train_df[cate_col].max()
        args.offsets.append(offset + 1)

    offset = int(offset + 1) # 도합
    args.offset = offset + 1 # 시계열 패딩 

    valid_df = test_df[test_df['answerCode'] != -1]
    print(train_df.columns)

    train_data = train_df.groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col != 'userID'])
        )

    valid_data = valid_df.groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col != 'userID'])
        )

    test_data = test_df.groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col != 'userID'])
        )
    return train_data, valid_data, test_data


class DKTDataset(Dataset):
    def __init__(self, data, args):
        # 현재 데이터는 userID 가 인덱스인데, 
        self.args = args
        self.data = data
        self.id2idx = {v:k for k, v in enumerate(self.data.index.unique())} 
        self.data.index = self.data.index.map(self.id2idx) 
        # userID 를 인덱싱함. 어차피 유저 아이디는 피처로 안들어가긴 함.
        # 판다스 인덱싱 접근 편의성 위해서.
        # 문제 자체가 어떤 유저가 들어왔을 때 임으로 userID는 일종의 인덱스역할임.

        
    def __getitem__(self, index):
        seq_len = len(self.data[index][0])
        cate_cols = [col.values.copy() + 1 for col in self.data[index] if col.name[-2:] == '_c' and \
            col.name != 'answerCode'] # 시계열 패딩 용 + 1. 이제 unknown 은 1 이 됨. 시계열 부족이 0
            
        cont_cols = [col.values.copy() for col in self.data[index] if col.name[-2:] != '_c' and \
            col.name != 'answerCode'] # cont 는 시계열 패딩을 어떻게..? # 마지막 최근 시계열 값으로 앞 시계열 채우겠습니다. 일단은.
        # print(cate_cols)
        answer = [col.values.copy() for col in self.data[index] if col.name == 'answerCode'][0].astype(np.float32)
        
        if seq_len >= self.args.max_seq_len:
            if cate_cols:
                for i, col in enumerate(cate_cols):
                    cate_cols[i] = col[-self.args.max_seq_len:]
            if cont_cols: # cont feature 가 없을 수도 있음.
                for i, col in enumerate(cont_cols):
                    cont_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int64) # Byte Tensor 를 씀
            answer = answer[-self.args.max_seq_len:]
        else:
            if cate_cols:
                for i in range(len(cate_cols)):
                    new_col = np.zeros(self.args.max_seq_len, np.int64)
                    new_col[-seq_len:] = cate_cols[i]
                    cate_cols[i] = new_col
            if cont_cols:
                for i in range(len(cont_cols)):
                    new_col = np.zeros(self.args.max_seq_len, dtype=np.float32)
                    # new_col[:seq_len] = cont_cols[i][0] # cont 의 시계열이 부족하면 가장 앞 데이터로 채움.
                    new_col[-seq_len:] = cont_cols[i]
                    cont_cols[i] = new_col
            mask = np.zeros(self.args.max_seq_len, dtype=np.int64)
            mask[-seq_len:] = 1
            new_answer = np.zeros(self.args.max_seq_len, dtype=np.float32)
            new_answer[-seq_len:] = answer
            answer = new_answer

        cate_cols = np.array(cate_cols).T.astype(np.int64)
        cont_cols = np.array(cont_cols).T.astype(np.float32)

        # return cate_cols, cont_cols, mask, target
        return cate_cols, cont_cols, mask, answer

    def __len__(self):
        return len(self.data)
    

class SAKTDataset(Dataset):
    def __init__(self, data, args):
        # 현재 데이터는 userID 가 인덱스인데, 
        self.args = args
        self.data = data
        self.id2idx = {v:k for k, v in enumerate(self.data.index.unique())} 
        self.data.index = self.data.index.map(self.id2idx) 
        # userID 를 인덱싱함. 어차피 유저 아이디는 피처로 안들어가긴 함.
        # 판다스 인덱싱 접근 편의성 위해서.
        # 문제 자체가 어떤 유저가 들어왔을 때 임으로 userID는 일종의 인덱스역할임.

        
    def __getitem__(self, index):
        seq_len = len(self.data[index][0])
        cate_cols = [col.values.copy() + 1 for col in self.data[index] if col.name[-2:] == '_c' and \
            col.name != 'answerCode'] # 시계열 패딩 용 + 1. 이제 unknown 은 1 이 됨. 시계열 부족이 0
            
        cont_cols = [col.values.copy() for col in self.data[index] if col.name[-2:] != '_c' and \
            col.name != 'answerCode'] # cont 는 시계열 패딩을 어떻게..? # 마지막 최근 시계열 값으로 앞 시계열 채우겠습니다. 일단은.
        # print(cate_cols)
        answer = [col.values.copy() for col in self.data[index] if col.name == 'answerCode'][0].astype(np.float32)
        
        if seq_len >= self.args.max_seq_len:
            if cate_cols:
                for i, col in enumerate(cate_cols):
                    cate_cols[i] = col[-self.args.max_seq_len:]
            if cont_cols: # cont feature 가 없을 수도 있음.
                for i, col in enumerate(cont_cols):
                    cont_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int64) # Byte Tensor 를 씀
            answer = answer[-self.args.max_seq_len:]
        else:
            if cate_cols:
                for i in range(len(cate_cols)):
                    new_col = np.zeros(self.args.max_seq_len, np.int64)
                    new_col[-seq_len:] = cate_cols[i]
                    cate_cols[i] = new_col
            if cont_cols:
                for i in range(len(cont_cols)):
                    new_col = np.zeros(self.args.max_seq_len, dtype=np.float32)
                    # new_col[:seq_len] = cont_cols[i][0] # cont 의 시계열이 부족하면 가장 앞 데이터로 채움.
                    new_col[-seq_len:] = cont_cols[i]
                    cont_cols[i] = new_col
            mask = np.zeros(self.args.max_seq_len, dtype=np.int64)
            mask[-seq_len:] = 1
            new_answer = np.zeros(self.args.max_seq_len, dtype=np.float32)
            new_answer[-seq_len:] = answer
            answer = new_answer

        cate_cols = np.array(cate_cols).T.astype(np.int64)
        cont_cols = np.array(cont_cols).T.astype(np.float32)

        # return cate_cols, cont_cols, mask, target
        return cate_cols, cont_cols, mask, answer

    def __len__(self):
        return len(self.data)