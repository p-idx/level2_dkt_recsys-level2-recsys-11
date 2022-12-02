#opt/ml/data에서 불러오기
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as ctb

import numpy as np
import random



# 데이터 로드 함수(train, test) from directory
def get_data(args):
    train_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv'))    # train + test(not -1)
    test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))    # test

    elim_col = ['user_correct_answer', 
                'user_total_answer',
                'interaction_c',
                'user_acc_c'
                ]
    for col in elim_col:
        train_data = train_data.drop(col, axis=1)
        test_data = test_data.drop(col, axis=1)
    
    cate_cols = [col for col in train_data.columns if col[-2:]== '_c']

    test = test_data[test_data.answerCode == -1]   # test last sequnece
    
    #테스트의 정답 컬럼을 제거
    test = test.drop('answerCode', axis=1)
    train = train_data
    return cate_cols, train, test


# 데이터 스플릿 함수
def data_split(train_data, args):
    if args.valid_exp:
        test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))    # 
        
        # drop unused coloumns
        elim_col = ['user_correct_answer', 
                    'user_total_answer',
                    'interaction_c',
                    'user_acc_c'
                    ]
        for col in elim_col:
            test_data = test_data.drop(col, axis=1)


        test_data = test_data.query('answerCode != -1')
        
        valid = test_data.groupby('userID').tail(args.valid_exp_n)
        print(f'valid.shape = {valid.shape}, valid.n_users = {valid.userID.nunique()}')
        train = train_data.drop(index = valid.index)

        ### has_time shuffle
        if args.has_time:
            X_train, y_train = time_shuffle(train)
        else:
            X_train = train.drop('answerCode', axis=1)
            y_train = train['answerCode']

        X_valid = valid.drop('answerCode', axis=1)
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



def time_shuffle(train):
    group = (train.groupby("userID").apply(lambda r: [r[name] for name in train.columns]))

    col = train.columns.tolist()
    for idx, c in enumerate(col):
        if c == 'answerCode':
            ac_idx = idx
            break
    
    print(f"Column '{col[ac_idx]}' taken out for y (has_time)")
    col.pop(ac_idx)

    X = [[] for _ in range(len(col))]
    Y = []

    # shuffle data grouped by users
    by_user = group.values
    random.shuffle(by_user)

    # realign shuffled train data to pd.DataFrame 
    for user in by_user:
        Y.extend(user.pop(ac_idx))

        for idx,feat in enumerate(user):
            X[idx].extend(feat)  

    X_train = pd.DataFrame({name:values for name,values in zip(col, X)})
    y_train = pd.DataFrame({'answerCode': Y})

    return X_train, y_train
