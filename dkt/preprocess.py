import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

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
        le_fe_train_df.to_csv(os.path.join(BASE_DATA_PATH, self.__class__.__name__, 'train_data.csv'), index=False)
        le_fe_test_df.to_csv(os.path.join(BASE_DATA_PATH, self.__class__.__name__, 'test_data.csv'), index=False)
        # le_fe_test_df.to_csv(os.path.join(f'/opt/ml/data/{self.__class__.__name__}', 'test_data.csv'), index=False)
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
        train_df['cont_ex'] = 1.0
        test_df['cont_ex'] = 1.0

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


class FE01(FeatureEngineer):
    def __str__(self):
        return \
            """시험 별로 최종 문항에 대한 수치형 피쳐 추가"""
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

        # train과 test를 merge하여 사용할 경우 결과가 조금 달라질 수 있다.
        # 큰 차이는 없을 것으로 보이는데, 일단 나눠서 진행한다.

        # 각 시험 속 문항번호를 수치형으로 만들어 추가한다.
        train_df['probnum'] = train_df['assessmentItemID'].apply(lambda x: int(x[-3:]))
        test_df['probnum'] = test_df['assessmentItemID'].apply(lambda x: int(x[-3:]))

        # 위 번호를 토대로 각 시험의 최종 문항을 피쳐로 추가한다.
        train_tmp = train_df.groupby('testId')
        train_tmp = train_tmp['probnum'].max()
        train_df['maxprob'] = train_df['testId'].map(train_tmp)
        test_tmp = test_df.groupby('testId')
        test_tmp = test_tmp['probnum'].max()
        test_df['maxprob'] = test_df['testId'].map(test_tmp)

        # 문항번호가 수치형으로 데이터에 들어갔으니, 기존 범주형 문항 번호는 삭제한다.
        train_df = train_df.drop('assessmentItemID', axis=1)
        test_df = test_df.drop('assessmentItemID', axis=1)

        # 수치형은 z정규화를 하기로 약속했다.
        nummean = train_df['probnum'].mean()
        numstd = train_df['probnum'].std()
        train_df['probnum'] = train_df['probnum'] - nummean / numstd
        nummean = test_df['probnum'].mean()
        numstd = test_df['probnum'].std()
        test_df['probnum'] = test_df['probnum'] - nummean / numstd

        nummean = train_df['maxprob'].mean()
        numstd = train_df['maxprob'].std()
        train_df['maxprob'] = train_df['maxprob'] - nummean / numstd
        nummean = test_df['maxprob'].mean()
        numstd = test_df['maxprob'].std()
        test_df['maxprob'] = test_df['maxprob'] - nummean / numstd



        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
            }
        )
        return train_df, test_df


class FE02(FeatureEngineer):
    def __str__(self):
        return \
            """시험 별로 최종 문항에 대한 범주형 피쳐 추가"""
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
        train_df['cont_ex'] = 1.0
        test_df['cont_ex'] = 1.0

        # train과 test를 merge하여 사용할 경우 결과가 조금 달라질 수 있다.
        # 큰 차이는 없을 것으로 보이는데, 일단 나눠서 진행한다.

        # 각 시험 속 문항번호를 수치형으로 만들어 추가한다.
        train_df['probnum'] = train_df['assessmentItemID'].apply(lambda x: int(x[-3:]))
        test_df['probnum'] = test_df['assessmentItemID'].apply(lambda x: int(x[-3:]))

        # 위 번호를 토대로 각 시험의 최종 문항을 피쳐로 추가한다.
        train_tmp = train_df.groupby('testId')
        train_tmp = train_tmp['probnum'].max()
        train_df['maxprob'] = train_df['testId'].map(train_tmp)
        test_tmp = test_df.groupby('testId')
        test_tmp = test_tmp['probnum'].max()
        test_df['maxprob'] = test_df['testId'].map(test_tmp)

        # 문항번호가 따로 데이터에 들어갔으니, 기존 범주형 문항 번호는 삭제한다.
        train_df = train_df.drop('assessmentItemID', axis=1)
        test_df = test_df.drop('assessmentItemID', axis=1)


        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
                'probnum' : 'probnum_c',
                'maxprob' : 'maxprob_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
                'probnum' : 'probnum_c',
                'maxprob' : 'maxprob_c',
            }
        )
        return train_df, test_df

class FE03(FeatureEngineer):
    def __str__(self):
        return \
            """FE00 + FE01"""
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
        train_df['cont_ex'] = 1.0
        test_df['cont_ex'] = 1.0

        # train과 test를 merge하여 사용할 경우 결과가 조금 달라질 수 있다.
        # 큰 차이는 없을 것으로 보이는데, 일단 나눠서 진행한다.

        # 각 시험 속 문항번호를 수치형으로 만들어 추가한다.
        train_df['probnum'] = train_df['assessmentItemID'].apply(lambda x: int(x[-3:]))
        test_df['probnum'] = test_df['assessmentItemID'].apply(lambda x: int(x[-3:]))

        # 위 번호를 토대로 각 시험의 최종 문항을 피쳐로 추가한다.
        train_tmp = train_df.groupby('testId')
        train_tmp = train_tmp['probnum'].max()
        train_df['maxprob'] = train_df['testId'].map(train_tmp)
        test_tmp = test_df.groupby('testId')
        test_tmp = test_tmp['probnum'].max()
        test_df['maxprob'] = test_df['testId'].map(test_tmp)

        # 문항번호가 수치형으로 데이터에 들어갔으니, 기존 범주형 문항 번호는 삭제한다.
        train_df = train_df.drop('assessmentItemID', axis=1)
        test_df = test_df.drop('assessmentItemID', axis=1)

        # 수치형은 z정규화를 하기로 약속했다.
        nummean = train_df['probnum'].mean()
        numstd = train_df['probnum'].std()
        train_df['probnum'] = train_df['probnum'] - nummean / numstd
        nummean = test_df['probnum'].mean()
        numstd = test_df['probnum'].std()
        test_df['probnum'] = test_df['probnum'] - nummean / numstd

        nummean = train_df['maxprob'].mean()
        numstd = train_df['maxprob'].std()
        train_df['maxprob'] = train_df['maxprob'] - nummean / numstd
        nummean = test_df['maxprob'].mean()
        numstd = test_df['maxprob'].std()
        test_df['maxprob'] = test_df['maxprob'] - nummean / numstd



        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
            }
        )
        return train_df, test_df

class FE04(FeatureEngineer):
    def __str__(self):
        return \
            """FE03에서 z정규화 제거"""
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
        train_df['cont_ex'] = 1.0
        test_df['cont_ex'] = 1.0

        # train과 test를 merge하여 사용할 경우 결과가 조금 달라질 수 있다.
        # 큰 차이는 없을 것으로 보이는데, 일단 나눠서 진행한다.

        # 각 시험 속 문항번호를 수치형으로 만들어 추가한다.
        train_df['probnum'] = train_df['assessmentItemID'].apply(lambda x: int(x[-3:]))
        test_df['probnum'] = test_df['assessmentItemID'].apply(lambda x: int(x[-3:]))

        # 위 번호를 토대로 각 시험의 최종 문항을 피쳐로 추가한다.
        train_tmp = train_df.groupby('testId')
        train_tmp = train_tmp['probnum'].max()
        train_df['maxprob'] = train_df['testId'].map(train_tmp)
        test_tmp = test_df.groupby('testId')
        test_tmp = test_tmp['probnum'].max()
        test_df['maxprob'] = test_df['testId'].map(test_tmp)

        # 문항번호가 수치형으로 데이터에 들어갔으니, 기존 범주형 문항 번호는 삭제한다.
        train_df = train_df.drop('assessmentItemID', axis=1)
        test_df = test_df.drop('assessmentItemID', axis=1)



        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
            }
        )
        return train_df, test_df
    
class FE05(FeatureEngineer):
    def __str__(self):
        return \
            """test_df FE test for my solution"""
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
        
        numeric_col = []
        test_user = test_df.userID.unique()
        train_user = train_df.userID.unique()

        
        fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        train_df['cont_ex'] = 1.0
        test_df['cont_ex'] = 1.0

        merged_df = pd.concat([train_df, test_df], axis=0)
        merged_df = merged_df.sort_values(['userID','Timestamp'])
        
        '''
        - shift를 진행하는 FE는 -1을 포함한 merged_df에 적용한다.
        - answerCode를 사용하는 FE는 -1 값을 빼뒀다가 mapping을 이용한다. 
        - 그 외의 FE는 
        ==> 하나의 코드 블럭에서 하나의 FE에 대해서만 적용할 수 있으므로, 문제가 생겼을 때 해결하기 쉬울 것
        '''
        
        ####################### Shift를 사용하는 Feature #######################
        # 유저가 문제를 푸는데 걸린 시간
        merged_df['shifted'] = merged_df.groupby(['userID','testId'])[['userID','Timestamp']].shift()['Timestamp']
        merged_df['solved_time'] = (merged_df['Timestamp'] - merged_df['shifted']).dt.total_seconds()
        merged_df = merged_df.drop('shifted', axis=1)
        
        numeric_col.append('solved_time') # 근데 이렇게 피쳐 생성 방법 별로 나누면 scaler 적용할 때 문제가 발생할 수 있음


        # 유저가 문제를 푸는데 걸린 시간 median
        
        ####################### answerCode를 사용하는 Feature #######################
        
        # -1인 값 분리
        test_df = merged_df.query('userID in @test_user')
        test_droped_df = test_df.query('answerCode == -1')
        merged_df = merged_df.query('answerCode != -1')
        test_df = test_df.query('answerCode != -1')
        
        # 시험지 문항 번호별 평균 정답률
        merged_df['prob_num'] = merged_df['assessmentItemID'].str[-3:] # assessmentItemID의 마지막 3글자가 문항 번호
        mean_val = merged_df.groupby('prob_num')['answerCode'].mean()
        merged_df['prob_num_mean'] = merged_df['prob_num'].map(mean_val)
        merged_df.drop('prob_num', axis=1, inplace=True)
        
        # test_droped_df는 -1인 행만 모아놓은 df
        test_droped_df['prob_num'] = test_droped_df['assessmentItemID'].str[-3:]
        test_droped_df['prob_num_mean'] = test_droped_df['prob_num'].map(mean_val)
        test_droped_df.drop('prob_num', axis=1, inplace=True)
        
        numeric_col.append('prob_num_mean')
        
        
        # 요일별 평균 정답률
        merged_df['days'] = merged_df['Timestamp'].dt.day_name()
        days_mean = merged_df.groupby('days')['answerCode'].mean()
        merged_df['days_mean'] = merged_df['days'].map(days_mean)
        merged_df.drop('days', axis=1, inplace=True)
        
        test_droped_df['days'] = test_droped_df['Timestamp'].dt.day_name()
        test_droped_df['days_mean'] = test_droped_df['days'].map(days_mean)
        test_droped_df.drop('days', axis=1, inplace=True)
        
        numeric_col.append('days_mean')
        
        # 시험지의 각 문항 별 평균 정답률
        asses_mean = merged_df.groupby('assessmentItemID')['answerCode'].mean()
        merged_df['asses_mean'] = merged_df['assessmentItemID'].map(asses_mean)
        
        test_droped_df['asses_mean'] = test_droped_df['assessmentItemID'].map(asses_mean)
        
        numeric_col.append('asses_mean')
        
        ####################### feature 구분 #######################
        
        # 수치형 feature 정규화
        scaler = StandardScaler()
        scaler.fit(merged_df[numeric_col])
        merged_df[numeric_col] = scaler.transform(merged_df[numeric_col])
        test_droped_df[numeric_col] = scaler.transform(test_droped_df[numeric_col])
        
        train_df = merged_df.query('userID in @train_user')
        test_df = merged_df.query('userID in @test_user')
        test_df = pd.concat([test_df, test_droped_df], axis=0) 
        test_df.sort_values(by=['userID', 'Timestamp'], inplace=True)
        
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

class FE06(FeatureEngineer):
    def __str__(self):
        return \
            """FE00 + 유저별 푼 문제수, 실력, 문제 별 난이도, 노출 정도, 태그 별 난이도, 노출 정도, 문제 풀리는데 걸리는 시간, 찍었는지 여부"""
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

        # TODO: merge한 DataFrame(merged)을 이용하여 feature engineering 진행 후, test_df에 새로 생성된 feature들을 merge해주는 방법
        fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        train_df['cont_ex'] = 1.0
        test_df['cont_ex'] = 1.0

        # merge를 하기 위해서 test_df에서 -1을 answerCode로 갖는 행을 제외한 df 생성
        test_tmp = test_df[test_df.answerCode != -1].reset_index()
        test_tmp = test_tmp.drop('index', axis=1)

        # merge후, interaction항 추가 해줌
        merged = pd.concat([train_df,test_tmp],axis=0)
        merged.sort_values(['userID','Timestamp'], inplace=True)
        merged = merged.reset_index().drop('index', axis=1)
        merged['interaction'] = merged.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']

        # 정답률을 계산하기 위한 percentile 함수 정의
        def percentile(s):
            return np.sum(s) / len(s)
        
        # grade를 나누기 위한 grade_map 함수 정의
        def grade_map(x : float):
            if x <= 0.4:
                return 0
            elif 0.4< x <0.8:
                return 1
            elif x >= 0.8:
                return 2

        ##### FE 시작 #####
        ##############################################################
        ## stu_groupby_merged : userID 이용한 FE ## : counts, user_grade 추가
        stu_groupby_merged = merged.groupby('userID').agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        })
        stu_groupby_merged.columns = ['counts', 'meanAnswerRate'] # groupby 집계, counts : 유저가 푼 문제의 개수
        test_df['counts'] = test_df['userID'].map(stu_groupby_merged['counts']) # test_df mapping
        
        stu_groupby_merged['user_grade'] = stu_groupby_merged['meanAnswerRate'].apply(grade_map) # 유저의 평균 정답률을 이용한 실력,등급 정의
        test_df['user_grade'] = test_df['userID'].map(stu_groupby_merged['user_grade']) # test_df mapping
        
        # merge : counts (유저가 푼 문제의 개수), user_grade (유저의 실력) 추가
        merged = merged.join(stu_groupby_merged.loc[:,['counts','user_grade']], how='left', on='userID')
        
        ##############################################################
        ## prob_groupby : assessmentItemID 이용한 FE ## 
        prob_groupby = merged.groupby('assessmentItemID').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        prob_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        prob_groupby['ass_grade'] = prob_groupby['meanAnswerRate'].apply(grade_map) # 문제 평균 정답률을 이용한 난이도 정의
        test_df['ass_grade'] = test_df['assessmentItemID'].map(prob_groupby['ass_grade']) # test_df mapping

        prob_solved_mean = prob_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 문제들은 몇 명에게 노출됐는가)
        prob_groupby['ass_solved'] = prob_groupby['numUsers'].apply(lambda x: int(x>prob_solved_mean)) # 문제가 많이 노출된 편인지, 아닌지 여부
        test_df['ass_solved'] = test_df['assessmentItemID'].map(prob_groupby['ass_solved']) # test_df mapping
        
        # merge : ass_grade, ass_solved 추가
        merged = merged.join(prob_groupby.loc[:,['ass_grade','ass_solved']], how='left', on='assessmentItemID')
        
        ##############################################################
        ## tag_groupby : KnowledgeTag 이용한 FE ## 
        tag_groupby = merged.groupby('KnowledgeTag').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        tag_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        tag_groupby['tag_grade'] = tag_groupby['meanAnswerRate'].apply(lambda x: grade_map(x)) # 태그 평균 정답률을 이용한 난이도 정의
        test_df['tag_grade'] = test_df['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_df mapping

        tag_solved_mean = tag_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 태그들은 몇 명에게 노출됐는가)
        tag_groupby['tag_solved'] = tag_groupby['numUsers'].apply(lambda x:int(x>tag_solved_mean)) # 태그가 많이 노출된 편인지, 아닌지 여부
        test_df['tag_solved'] = test_df['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_df mapping

        # merge : tag_grade, tag_solced 추가
        merged = merged.join(tag_groupby.loc[:,['tag_grade','tag_solved']], how='left', on='KnowledgeTag')
        
        ##############################################################
        ## Timestamp 이용한 FE ##
        # Timestamp 차이를 통해 이번문제를 푸는데 얼마나 걸렸는가에 대해 diff 변수 생성하기 (미션1에서 잘못된 부분 수정함)
        merged['Timestamp'] = pd.to_datetime(merged['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        diff = merged.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff.fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        merged['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간
        merged['mark_randomly'] = merged['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주

        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        diff = test_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff.fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        test_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간
        test_df['mark_randomly'] = test_df['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주

        ## test_df에서 우리가 예측해야할 sequence는 모두 elapsed가 0.0이다 (문제를 풀었다는 정보가 없기 때문에)
        ## 따라서 그 부분을 조금이라도 대치해주기 위해서 test_last_sequence라는 df를 생성 (-1인 것들 분리)
        test_last_sequence = test_df.loc[test_df['answerCode'] == -1, :]
        test_df = test_df.loc[test_df['answerCode'] != -1, :]

        ## 해당 문제의 median elapsed time으로 대치해주기
        test_last_sequence['elapsed'] = test_last_sequence['assessmentItemID'].map(merged.groupby(['assessmentItemID'])['elapsed'].median())
        
        ## 다시 concat하고, index는 건드리지 않았기 때문에 index 기준으로 정렬해주기 (원상복구됨)
        test_df = pd.concat([test_df, test_last_sequence], axis=0)
        test_df.sort_index(inplace=True)
        
        train_df = merged # train_df를 merge한 걸로 덮어쓰기

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
                'user_grade' : 'user_grade_c',
                'ass_grade' : 'ass_grade_c',
                'ass_solved' : 'ass_solved_c',
                'tag_grade' : 'tag_grade_c',
                'tag_solved' : 'tag_solved_c',
                'mark_randomly' : 'mark_randomly_c'
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
                'user_grade' : 'user_grade_c',
                'ass_grade' : 'ass_grade_c',
                'ass_solved' : 'ass_solved_c',
                'tag_grade' : 'tag_grade_c',
                'tag_solved' : 'tag_solved_c',
                'mark_randomly' : 'mark_randomly_c'
            }
        )
        return train_df, test_df


def main():
    base_train_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'train_data.csv'))
    base_test_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'test_data.csv'))

    # 클래스 생성 후 여기에 번호대로 추가해주세요.
    # FE00(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE01(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE02(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE03(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE04(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE05(BASE_DATA_PATH, base_train_df, base_test_df).run()
    FE06(BASE_DATA_PATH, base_train_df, base_test_df).run()

if __name__=='__main__':
    main()
