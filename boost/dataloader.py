#opt/ml/data에서 불러오기
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import catboost as ctb
from sklearn.utils import shuffle

import numpy as np



# 데이터 로드 함수(train, test) from directory
def get_data(args):
    train_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv'))    # train + test(not -1)
    test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))    # test
    sub_test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'sub_test_data.csv'))    # sub test
    # train_data = train_data.drop(['interaction_c'], axis=1)
    # test_data = test_data.drop(['interaction_c'], axis=1)
    # train_data 중복 제거
    
    cate_cols = [col for col in train_data.columns if col[-2:]== '_c']

    test = test_data[test_data.answerCode == -1]   # test last sequnece
    # sub_test_data = sub_test_data.drop(['answerCode'], axis=1)
    #테스트의 정답 컬럼을 제거
    test = test.drop(['answerCode'], axis=1)
    train = train_data
    return cate_cols, train, test, sub_test_data


# 데이터 스플릿 함수
def data_split(train_data, args):
    if args.valid_exp:
        test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))    # test
        test_data = test_data.query('answerCode != -1')
        # test_data = test_data.drop(['interaction_c'], axis=1)
        if args.is_new:
            total = train_data.groupby('userID').tail(args.valid_exp_n)
            train, valid = train_test_split(total, test_size=0.2)
        else:
            valid = train_data.groupby('userID').tail(args.valid_exp_n)
            train = train_data.drop(index = valid.index)

        print(f'valid.shape = {valid.shape}, valid.n_users = {valid.userID.nunique()}')
        # 기존
        # test_user = test_data['userID'].unique()
        # valid = train_data.query('userID in @test_user').groupby('userID').tail(args.valid_exp_n)
        # train = train_data.drop(index = valid.index)
        print(f'train.shape = {train_data.shape}')
        print(f'ideal.shape = {len(train_data) - len(valid)}')
        print(f'valid.shape = {valid.shape}, valid.n_users = {valid.userID.nunique()}')
        print(f'after train.shape = {train.shape}')
        X_train = train.drop('answerCode', axis=1)
        X_valid = valid.drop('answerCode', axis=1)
        y_train = train['answerCode']
        y_valid = valid['answerCode']


    else:
        X = train_data.drop(['answerCode'], axis=1)
        y = train_data['answerCode']

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=args.ratio, # 일단 이 정도로 학습해서 추이 확인
            shuffle=True,
        )
    
    return X_train, X_valid, y_train, y_valid

def option1_train_test_split(train_data, args):

    users = list(zip(train_data["userID"].value_counts().index, train_data["userID"].value_counts()))
    random.shuffle(users)

    max_train_data_len = args.ratio * len(train_data)
    sum_of_train_data = 0
    user_ids = []

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)

    train = train_data[train_data["userID"].isin(user_ids)]
    valid = train_data[train_data["userID"].isin(user_ids) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    # test = test[test["userID"] != test["userID"].shift(-1)]
    X_train = train.drop('answerCode', axis=1)
    X_valid = valid.drop('answerCode', axis=1)
    y_train = train['answerCode']
    y_valid = valid['answerCode']


    return X_train, X_valid, y_train, y_valid

def option1_5fold_train_test_split(train_data):

    users = list(zip(train_data["userID"].value_counts().index, train_data["userID"].value_counts()))
    random.shuffle(users)

    val_data_len = len(train_data)//5
    sum_of_train_data = 0
    user_ids1 = []
    user_ids2 = []
    user_ids3 = []
    user_ids4 = []
    user_ids5 = []

    for user_id, count in users:
        sum_of_train_data += count
        if sum_of_train_data < val_data_len:
            user_ids1.append(user_id)
        elif sum_of_train_data < val_data_len*2:
            user_ids2.append(user_id)
        elif sum_of_train_data < val_data_len*3:
            user_ids3.append(user_id)
        elif sum_of_train_data < val_data_len*4:
            user_ids4.append(user_id)
        else:
            user_ids5.append(user_id)   
        
    train = train_data[train_data["userID"].isin(user_ids1)]
    valid = train_data[train_data["userID"].isin(user_ids1) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    # test = test[test["userID"] != test["userID"].shift(-1)]
    X_train = train.drop('answerCode', axis=1)
    X_valid = valid.drop('answerCode', axis=1)
    y_train = train['answerCode']
    y_valid = valid['answerCode']

    
    return [user_ids1, user_ids2, user_ids3, user_ids4, user_ids5]
