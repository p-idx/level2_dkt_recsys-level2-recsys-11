import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# from args import parse_args
BASE_DATA_PATH = '/opt/ml/data'

def convert_time(s: str) -> int:
    timestamp = time.mktime(
        datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
    )
    return int(timestamp)


class FeatureEngineer:
    def __init__(self, base_path, base_train_df, base_test_df):
        self.base_path = base_path
        self.base_train_df = base_train_df
        self.base_test_df = base_test_df

    def __label_encoding(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> pd.DataFrame:
        cate_cols = [col for col in train_df.columns if col[-2:] == '_c']
        not_cate_cols = [col for col in train_df.columns if col not in cate_cols]

        # train_df 에 unknown 용 np.nan 을 각각 추가해준다.
        # 피처 중 np.nan 자체가 없는 피처가 있을 수 있으므로(노결측치)
        train_df.loc[len(train_df)] = [np.nan for _ in range(len(train_df.columns))]

        or_enc = OrdinalEncoder().set_params(encoded_missing_value=np.nan)
        or_enc.fit(train_df.drop(not_cate_cols, axis=1))

        train_np = or_enc.transform(train_df.drop(not_cate_cols, axis=1)) # not_cate_cols 우하하게
        test_np = or_enc.transform(test_df.drop(not_cate_cols, axis=1))

        offset = 0
        train_df[cate_cols] = train_np + 1 # np.nan + 1 = np.nan 임으로 이게 가능하다.
        test_df[cate_cols] = test_np + 1
        for cate_name in cate_cols:
            train_df[cate_name] += offset
            test_df[cate_name] += offset
            offset = train_df[cate_name].max()

        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)

        train_df[cate_cols + ['userID', 'answerCode']] =\
            train_df[cate_cols + ['userID', 'answerCode']].astype(np.int64)
        test_df[cate_cols + ['userID', 'answerCode']] =\
            test_df[cate_cols + ['userID', 'answerCode']].astype(np.int64)

        train_df.iloc[-1, 0] = offset + 1 # 1은 0
        train_df.iloc[-1, 1] = len(cate_cols)
        train_df.iloc[-1, 2] = len(not_cate_cols) - 2 # userID, answerCode 제외
        return train_df, test_df # np.nan 용 행 제거


    def run(self):
        print(f'[{self.__class__.__name__}] {self}')
        print(f'[{self.__class__.__name__}] preprocessing start...')

        if not os.path.exists(os.path.join(self.base_path, self.__class__.__name__)):
            os.mkdir(os.path.join(self.base_path, self.__class__.__name__))

        print(f'[{self.__class__.__name__}] feature engineering...')
        fe_train_df, fe_test_df = self.feature_engineering(self.base_train_df, self.base_test_df)

        fe_train_df = fe_train_df.drop(['Timestamp'], axis=1)
        fe_test_df = fe_test_df.drop(['Timestamp'], axis=1)

        print(f'[{self.__class__.__name__}] label encoding...')
        le_fe_train_df, le_fe_test_df = self.__label_encoding(fe_train_df, fe_test_df)
         
        print(f'[{self.__class__.__name__}] save...')
        le_fe_train_df.to_csv(os.path.join(f'/opt/ml/data/{self.__class__.__name__}', 'train_data.csv'), index=False)
        le_fe_test_df.to_csv(os.path.join(f'/opt/ml/data/{self.__class__.__name__}', 'test_data.csv'), index=False)
        print(f'[{self.__class__.__name__}] done.')


    def feature_engineering(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    

# baseline EDA
class FE00(FeatureEngineer):
    def __str__(self):
        return \
            """유저의 시험 별로 한 칸씩 내려 이전 시험문제를 맞추었는지에 대한 feature 추가"""
    def feature_engineering(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> pd.DataFrame:
        #################################
        # 완전 베이스 데이터로 시작합니다.
        #
        # Timestamp 컬럼은 이후 버려집니다. 버리실 필요 없습니다.
        # userID, answerCode 는 수정할 수 없습니다. test 의 -1 로 되어있는 부분 그대로 가져갑니다. (컬럼 위치 변경은 가능합니다.)
        # 새 카테고리 컬럼을 만들 때, 결측치가 생길 시 np.nan 으로 채워주세요. *'None', -1 등 불가
        # 새 컨티뉴어스 컬럼을 만들 때, 결측치가 생길 시 imputation 해주세요. ex) mean... etc. *np.nan은 불가
        # tip) imputation 이 어렵다면, 이전 대회의 age 범주화 같은 방법을 사용해 카테고리 컬럼으로 만들어 주세요.
        #################################

        # TODO: merge 하면 그대로 eda 진행 후 test_df 따로 떼주세요. 하단은 merge 없는 예
        fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        train_df['cont_ex'] = 0.0
        test_df['cont_ex'] = 0.0

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
            }
        )
        return train_df, test_df


def main():
    base_train_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'train_data.csv'))
    base_test_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'test_data.csv'))

    # 클래스 생성 후 여기에 번호대로 추가해주세요.
    FE00(BASE_DATA_PATH, base_train_df, base_test_df).run()


if __name__=='__main__':
    main()
