#opt/ml/data에서 불러오기
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as ctb

import numpy as np



# 데이터 로드 함수(train, test) from directory
def get_data(args):
    train_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv'))    # train + test(not -1)
    test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))    # test

    if 'drop_features' in args:
        if args.drop_features != []:
            train_data = train_data.drop(args.drop_features, axis=1)
            test_data = test_data.drop(args.drop_features, axis=1)

    cate_cols = [col for col in train_data.columns if col[-2:]== '_c']

    test = test_data[test_data.answerCode == -1]   # test last sequnece

    #테스트의 정답 컬럼을 제거
    test = test.drop('answerCode', axis=1)
    train = train_data
    return cate_cols, train, test


# 데이터 스플릿 함수
def data_split(train_data, test_data, args):
    if args.valid_exp:
        # test_user = test_data['userID'].unique()
        # valid = train_data.query('userID in @test_user').groupby('userID').tail(args.valid_exp_n)

        # print(f'train.shape = {train_data.shape}')
        # print(f'ideal.shape = {len(train_data) - len(valid)}')
        # print(f'valid.shape = {valid.shape}, valid.n_users = {valid.userID.nunique()}')
        # train = train_data.drop(index = valid.index)
        # print(f'after train.shape = {train.shape}')
        # X_train = train.drop('answerCode', axis=1)
        # X_valid = valid.drop('answerCode', axis=1)

        # y_train = train['answerCode']
        # y_valid = valid['answerCode']
        test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))    # test
        test_data = test_data.query('answerCode != -1')
        valid = test_data.groupby('userID').tail(args.valid_exp_n)
        train = train_data.drop(index = valid.index)
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