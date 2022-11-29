import os
from datetime import datetime

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple


def _label_encoding(
    args, 
    train_df:pd.DataFrame, 
    test_df:pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # train, test 의 Timestamp drop.
    train_df = train_df.drop(columns=['Timestamp'], axis=1)
    test_df = test_df.drop(columns=['Timestamp'], axis=1)

    if args.userid :
        train_df = pd.concat([train_df, test_df[test_df['answerCode'] != -1]])
        train_df['user_id_c'] = train_df['userID'].copy() 
        test_df['user_id_c'] = test_df['userID'].copy()
        # 이렇게 따로 빼면 되네 어차피 유저아이디로 그룹바이 해야하는데.
        # userid 를 피처로 사용하면 테스트 데이터도 완전히 합쳐서 학습해야 의미있음.
        # EDA 를 합쳐서 하느냐, 학습을 합쳐서 하느냐는 별개의 문제.

    cate_cols = [col for col in train_df.columns if col[-2:] == '_c']
    args.cate_num = len(cate_cols)
    args.cont_num = len(train_df.columns) - len(cate_cols) - 2 # 유저아이디, 앤서코드 - 2

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    oe = oe.fit(train_df[cate_cols])

    train_df[cate_cols] = oe.transform(train_df[cate_cols]) + 1 # np.nan 분리용
    test_df[cate_cols] = oe.transform(test_df[cate_cols]) + 1   

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    if args.emb_separate:
        offsets = []   
        for cate_col in cate_cols:
            offsets.append(train_df[cate_col].max() + 2) # 시계열 패딩
        args.offsets = offsets 
    else:
        offset = 0
        for cate_col in cate_cols:
            train_df[cate_col] += offset
            test_df[cate_col] += offset
            offset = train_df[cate_col].max()
        args.offset = int(offset + 2) # 시계열 패딩

    train_df[cate_cols] = train_df[cate_cols].astype(np.int64)
    test_df[cate_cols] = test_df[cate_cols].astype(np.int64)

    return train_df, test_df



def load_data(args, is_train: bool):
    if args.fe_num == 'base':
        train_df = pd.read_csv(os.path.join(args.data_dir, 'train_data.csv'))
        test_df = pd.read_csv(os.path.join(args.data_dir, 'train_data.csv'))
    else:
        train_df = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv'))
        test_df = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv'))

    train_df, test_df = _label_encoding(args, train_df, test_df)
    # 조교님 베이스라인은 asset 에 저장 후 test 를 적용하지만
    # asset 에 저장된 걸 불러오나 매번 새로 인코딩하나 시간차이가 크지 않은듯.

    if is_train:
        return train_df.groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col != 'userID'])
        )
    else:
        return test_df.groupby('userID').apply(
            lambda df: tuple([df[col] for col in df.columns if col != 'userID'])
        )

        
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
            mask = np.ones(self.args.max_seq_len, dtype=np.int16) # Byte Tensor 를 씀
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
    