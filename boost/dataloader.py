#opt/ml/data에서 불러오기
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import catboost as ctb

import numpy as np



# 데이터 로드 함수(train, test) from directory
def get_data(args):
    train_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(args.data_dir, f'FE{args.fe_num}', 'test_data.csv'))

    # merge/분리
    df = train_data.merge(test_data, how='outer')

    cate_cols = [col for col in df.columns if col[-2:]== '_c']

    test = df[df.answerCode == -1]
    train = df[df.answerCode != -1]
    #테스트의 정답 컬럼을 제거
    test = test.drop('answerCode', axis=1)
    return cate_cols, train, test


# 데이터 스플릿 함수
def data_split(train_data, ratio):
    X = train_data.drop(['answerCode'], axis=1)
    y = train_data['answerCode']

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=ratio, # 일단 이 정도로 학습해서 추이 확인
        shuffle=True,
    )
    return X_train, X_valid, y_train, y_valid