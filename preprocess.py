import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")
from typing import Tuple

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

BASE_DATA_PATH = '/opt/ml/level2_dkt_recsys-level2-recsys-11/data/'


class FeatureEngineer:
    def __init__(self, base_path, base_train_df, base_test_df, is_leakage=False):
        self.base_path = base_path
        self.base_train_df = base_train_df
        self.base_test_df = base_test_df
        self.is_leakage = is_leakage

    def __label_encoding(
        self, 
        train_df:pd.DataFrame, 
        test_df:pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train_df.loc[len(train_df)] = np.nan
        cate_cols = [col for col in train_df.columns if col[-2:] == '_c']

        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe = oe.fit(train_df[cate_cols])

        train_df[cate_cols] = oe.transform(train_df[cate_cols]) + 1 # np.nan 분리용
        test_df[cate_cols] = oe.transform(test_df[cate_cols]) + 1   

        train_df = train_df.fillna(0)
        test_df = test_df.fillna(0)

        train_df = train_df[:-1]
        
        offset = 0
        for cate_col in cate_cols:
            train_df[cate_col] += offset
            test_df[cate_col] += offset
            offset = train_df[cate_col].max()
        

        offset = int(offset + 1)


        train_df[cate_cols + ['userID', 'answerCode']] = \
            train_df[cate_cols + ['userID', 'answerCode']].astype(np.int64)
        test_df[cate_cols + ['userID', 'answerCode']] = \
            test_df[cate_cols + ['userID', 'answerCode']].astype(np.int64)

        return train_df, test_df, offset


    def run(self):
        print(f'[{self.__class__.__name__}] {self}')
        print(f'[{self.__class__.__name__}] preprocessing start...')

        if not os.path.exists(os.path.join(self.base_path, self.__class__.__name__)):
            os.mkdir(os.path.join(self.base_path, self.__class__.__name__))

        print(f'[{self.__class__.__name__}] feature engineering...')
        fe_train_df, fe_test_df = self.feature_engineering(self.base_train_df, self.base_test_df)

        fe_train_df = fe_train_df.drop(['Timestamp'], axis=1)
        fe_test_df = fe_test_df.drop(['Timestamp'], axis=1)

        print(f'[{self.__class__.__name__}] save csv...')
        fe_train_df.to_csv(os.path.join(BASE_DATA_PATH, self.__class__.__name__, 'train_data.csv'), index=False)
        fe_test_df.to_csv(os.path.join(BASE_DATA_PATH, self.__class__.__name__, 'test_data.csv'), index=False)

        print(f'[{self.__class__.__name__}] columns')
        print(fe_train_df.columns)
        print(f'[{self.__class__.__name__}] label encoding...')
        le_train_df, le_test_df, offset = self.__label_encoding(fe_train_df, fe_test_df)

        print(f'[{self.__class__.__name__}] save le csv...')
        le_train_df.to_csv(os.path.join(BASE_DATA_PATH, self.__class__.__name__, 'le_train_data.csv'), index=False)
        le_test_df.to_csv(os.path.join(BASE_DATA_PATH, self.__class__.__name__, 'le_test_data.csv'), index=False)
        
        with open(os.path.join(BASE_DATA_PATH, self.__class__.__name__, 'offset.txt'), 'w') as f:
            f.write(f'offset={offset}\n')

        print(f'[{self.__class__.__name__}] done.')


    def feature_engineering(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


# baseline EDA
class FE00(FeatureEngineer):
    def __str__(self):
        return \
            """아무것도 처리하지 않은 상태"""
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        # train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        # test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    

        # train_df['cont_ex'] = 0.0
        # test_df['cont_ex'] = 0.0

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                # 'interaction' : 'interaction_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                # 'interaction' : 'interaction_c',
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
            """문제 풀이에 걸린 시간 / 문항 번호별 평균 정답률 / 요일별 평균 정답률 / 각 문항별 평균 정답률"""
    def feature_engineering(self, train_df:pd.DataFrame, test_df:pd.DataFrame) -> pd.DataFrame:
                
        '''
        FE 방법
        - shift를 진행하는 FE는 -1을 포함한 merged_df에 적용한다.
        - answerCode를 사용하는 FE는 -1 값을 빼뒀다가 mapping을 이용한다. 
        - 그 외의 FE는 
        ==> 하나의 코드 블럭에서 하나의 FE에 대해서만 적용할 수 있으므로, 문제가 생겼을 때 해결하기 쉬울 것
        '''
        
        fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        
        numeric_col = [] # 정규화 적용할 column 추가
        test_user = test_df.userID.unique()
        train_user = train_df.userID.unique()
        
        train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        train_df['cont_ex'] = 1.0 # numeric 보험용
        test_df['cont_ex'] = 1.0

        merged_df = pd.concat([train_df, test_df], axis=0)
        merged_df = merged_df.sort_values(['userID','Timestamp'])

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

        numeric_col = []

        #### 1. train_df, test_df 에서 interaction, elapsed 구해놓기 ####
        # train_df = pd.read_csv('../data/train_data.csv')
        # test_df = pd.read_csv('../data/test_data.csv')
        train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode'].fillna(-1)
        test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode'].fillna(-1)

        diff = train_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        train_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간

        diff = test_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        test_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간        
        
        numeric_col.append('elapsed')

        #### 2. test_df 에서 test_tmp, test_last_sequence 떼어내기 ####
        # - test_tmp : not -1
        # - test_last_sequence : only -1
        # merge를 하기 위해서 test_df에서 -1을 answerCode로 갖는 행을 제외한 df 생성
        test_tmp = test_df[test_df.answerCode != -1]
        test_last_sequence = test_df[test_df.answerCode == -1]

        #### 3. train_df + test_tmp = merged 로 concat하기 ####
        # merge후, interaction항 추가 해줌
        merged = pd.concat([train_df,test_tmp],axis=0)
        merged.sort_values(['userID','Timestamp'], inplace=True)
        merged = merged.reset_index(drop=True) #.drop('index', axis=1)
        
        #### 4. merged 기준으로 FE를 진행, test_tmp와 test_last_sequence에도 각각의 정보(userID, assessmentItemID)를 이용해서 mapping ####
        ## 1. stu_groupby_merged : counts, user_grade 추가 ##
        stu_groupby_merged = merged.groupby('userID').agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        })
        stu_groupby_merged.columns = ['counts', 'meanAnswerRate'] # groupby 집계, counts : 유저가 푼 문제의 개수
        merged['counts'] = merged['userID'].map(stu_groupby_merged['counts']) # merged mapping
        test_tmp['counts'] = test_tmp['userID'].map(stu_groupby_merged['counts']) # test_tmp mapping
        test_last_sequence['counts'] = test_last_sequence['userID'].map(stu_groupby_merged['counts']) # test_last_sequence mapping

        stu_groupby_merged['user_grade'] = stu_groupby_merged['meanAnswerRate'].apply(grade_map) # 유저의 평균 정답률을 이용한 실력,등급 정의
        merged['user_grade'] = merged['userID'].map(stu_groupby_merged['user_grade']) # merged mapping
        test_tmp['user_grade'] = test_tmp['userID'].map(stu_groupby_merged['user_grade']) # test_tmp mapping
        test_last_sequence['user_grade'] = test_last_sequence['userID'].map(stu_groupby_merged['user_grade']) # test_last_sequence mapping

        numeric_col.append('counts')

        ## 2. prob_groupby : assessmentItemID 이용한 FE ## 
        prob_groupby = merged.groupby('assessmentItemID').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        prob_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        prob_groupby['ass_grade'] = prob_groupby['meanAnswerRate'].apply(grade_map) # 문제 평균 정답률을 이용한 난이도 정의
        merged['ass_grade'] = merged['assessmentItemID'].map(prob_groupby['ass_grade']) # merged mapping
        test_tmp['ass_grade'] = test_tmp['assessmentItemID'].map(prob_groupby['ass_grade']) # test_tmp mapping
        test_last_sequence['ass_grade'] = test_last_sequence['assessmentItemID'].map(prob_groupby['ass_grade']) # test_last_sequence mapping

        prob_solved_mean = prob_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 문제들은 몇 명에게 노출됐는가)
        prob_groupby['ass_solved'] = prob_groupby['numUsers'].apply(lambda x: int(x>prob_solved_mean)) # 문제가 많이 노출된 편인지, 아닌지 여부
        merged['ass_solved'] = merged['assessmentItemID'].map(prob_groupby['ass_solved']) # merged mapping
        test_tmp['ass_solved'] = test_tmp['assessmentItemID'].map(prob_groupby['ass_solved']) # test_tmp mapping
        test_last_sequence['ass_solved'] = test_last_sequence['assessmentItemID'].map(prob_groupby['ass_solved']) # test_last_sequence mapping 

        ## 3. tag_groupby : KnowledgeTag 이용한 FE ## 
        tag_groupby = merged.groupby('KnowledgeTag').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        tag_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        tag_groupby['tag_grade'] = tag_groupby['meanAnswerRate'].apply(lambda x: grade_map(x)) # 태그 평균 정답률을 이용한 난이도 정의
        merged['tag_grade'] = merged['KnowledgeTag'].map(tag_groupby['tag_grade']) # merged mapping
        test_tmp['tag_grade'] = test_tmp['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_tmp mapping
        test_last_sequence['tag_grade'] = test_last_sequence['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_last_sequence mapping

        tag_solved_mean = tag_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 태그들은 몇 명에게 노출됐는가)
        tag_groupby['tag_solved'] = tag_groupby['numUsers'].apply(lambda x:int(x>tag_solved_mean)) # 태그가 많이 노출된 편인지, 아닌지 여부
        merged['tag_solved'] = merged['KnowledgeTag'].map(tag_groupby['tag_solved']) # merged mapping
        test_tmp['tag_solved'] = test_tmp['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_tmp mapping
        test_last_sequence['tag_solved'] = test_last_sequence['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_last_sequence mapping

        #### 4-1 : merged 에서 elapsed가 0인 문제들 대치 해주기
        # - 보통 시험의 마지막 문제는 elapsed가 0이다. (그 문제를 풀고 끝나기 때문에, 얼마나 걸렸는지 알 수가 없고, 그렇기 때문에 그 값을 0으로 대치하는 느낌)
        # - **이 값들을 효과적으로 대치할 수 있으면, test_last_sequence에 elapsed를 효과적으로 전달할 수 있기 때문에 미리 진행**
        # - 우선 merged에서, 시간이 900초 이상 (15분 이상)소요된 풀이시간은 모두 900초로 대치해주자 (대략 31421건)
        merged['elapsed'] = merged['elapsed'].apply(lambda x : 900 if x > 900 else x)

        # - elapsed가 0이 아닌것과 0인것을 나눠서 일단 쪼개고 (인덱스는 건들지 말자), elapsed가 0인 data frame에 유저별 문제풀이 시간의 중앙값으로 대치하고 다시 합쳐주자
        # - 합칠때는 concat으로 위아래로 붙인다음에 index 기준 정렬
        merged_elapsed_not0 = merged[merged.elapsed != 0]
        merged_elapsed_0 = merged[merged.elapsed == 0]
        merged_elapsed_0['elapsed'] = merged_elapsed_0['userID'].map(merged.groupby('userID')['elapsed'].median())

        merged = pd.concat([merged_elapsed_not0,merged_elapsed_0], axis=0)
        merged = merged.sort_index()
        
        # - 이제 test_last_sequence에 있는 elapsed가 0인 애들은, 다른 사람들은 그 문제를 푸는데 얼마나 걸렸는지를 기준으로 대치할 수 있게 되었다
        # - 그러고 합치자 (test_tmp랑 test_last_sequence랑)
        test_last_sequence['elapsed'] = test_last_sequence['assessmentItemID'].map(merged.groupby('assessmentItemID')['elapsed'].median())
        test_df = pd.concat([test_tmp, test_last_sequence], axis=0).sort_index()

        # - 이제 elapsed가 잘 대치 되어있기 때문에, mark_randomly feature를 만들 수 있다.
        merged['mark_randomly'] = merged['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주
        test_df['mark_randomly'] = test_df['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주

        # 수치형 feature 정규화
        # scaler = StandardScaler()
        # scaler.fit(merged[numeric_col])
        # merged[numeric_col] = scaler.transform(merged[numeric_col])
        # test_df[numeric_col] = scaler.transform(test_df[numeric_col])

        # data leakage 허용 : merged가 train data 의 역할을 하자
        train_df = merged

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


class FE07(FeatureEngineer):
    def __str__(self):
        return \
            """FE05 + FE06"""
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
        
        numeric_col = []

        #### 1. train_df, test_df 에서 interaction, elapsed 구해놓기 ####
        # train_df = pd.read_csv('../data/train_data.csv')
        # test_df = pd.read_csv('../data/test_data.csv')
        train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode'].fillna(-1)
        test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode'].fillna(-1)

        diff = train_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        train_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간

        diff = test_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        test_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간        
        
        #### 2. test_df 에서 test_tmp, test_last_sequence 떼어내기 ####
        # - test_tmp : not -1
        # - test_last_sequence : only -1
        # merge를 하기 위해서 test_df에서 -1을 answerCode로 갖는 행을 제외한 df 생성
        test_tmp = test_df[test_df.answerCode != -1]
        test_last_sequence = test_df[test_df.answerCode == -1]

        #### 3. train_df + test_tmp = merged 로 concat하기 ####
        # merge후, interaction항 추가 해줌
        merged = pd.concat([train_df,test_tmp],axis=0)
        merged.sort_values(['userID','Timestamp'], inplace=True)
        merged = merged.reset_index(drop=True) #.drop('index', axis=1)
        
        #### 4. merged 기준으로 FE를 진행, test_tmp와 test_last_sequence에도 각각의 정보(userID, assessmentItemID)를 이용해서 mapping ####
        ## 1. stu_groupby_merged : counts, user_grade 추가 ##
        stu_groupby_merged = merged.groupby('userID').agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        })
        stu_groupby_merged.columns = ['counts', 'meanAnswerRate'] # groupby 집계, counts : 유저가 푼 문제의 개수
        merged['counts'] = merged['userID'].map(stu_groupby_merged['counts']) # merged mapping
        test_tmp['counts'] = test_tmp['userID'].map(stu_groupby_merged['counts']) # test_tmp mapping
        test_last_sequence['counts'] = test_last_sequence['userID'].map(stu_groupby_merged['counts']) # test_last_sequence mapping

        stu_groupby_merged['user_grade'] = stu_groupby_merged['meanAnswerRate'].apply(grade_map) # 유저의 평균 정답률을 이용한 실력,등급 정의
        merged['user_grade'] = merged['userID'].map(stu_groupby_merged['user_grade']) # merged mapping
        test_tmp['user_grade'] = test_tmp['userID'].map(stu_groupby_merged['user_grade']) # test_tmp mapping
        test_last_sequence['user_grade'] = test_last_sequence['userID'].map(stu_groupby_merged['user_grade']) # test_last_sequence mapping

        ## 2. prob_groupby : assessmentItemID 이용한 FE ## 
        prob_groupby = merged.groupby('assessmentItemID').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        prob_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        prob_groupby['ass_grade'] = prob_groupby['meanAnswerRate'].apply(grade_map) # 문제 평균 정답률을 이용한 난이도 정의
        merged['ass_grade'] = merged['assessmentItemID'].map(prob_groupby['ass_grade']) # merged mapping
        test_tmp['ass_grade'] = test_tmp['assessmentItemID'].map(prob_groupby['ass_grade']) # test_tmp mapping
        test_last_sequence['ass_grade'] = test_last_sequence['assessmentItemID'].map(prob_groupby['ass_grade']) # test_last_sequence mapping

        prob_solved_mean = prob_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 문제들은 몇 명에게 노출됐는가)
        prob_groupby['ass_solved'] = prob_groupby['numUsers'].apply(lambda x: int(x>prob_solved_mean)) # 문제가 많이 노출된 편인지, 아닌지 여부
        merged['ass_solved'] = merged['assessmentItemID'].map(prob_groupby['ass_solved']) # merged mapping
        test_tmp['ass_solved'] = test_tmp['assessmentItemID'].map(prob_groupby['ass_solved']) # test_tmp mapping
        test_last_sequence['ass_solved'] = test_last_sequence['assessmentItemID'].map(prob_groupby['ass_solved']) # test_last_sequence mapping 

        # FE05 내용 추가
        # 시험지 문항 번호별 평균 정답률
        merged['prob_num'] = merged['assessmentItemID'].str[-3:] # assessmentItemID의 마지막 3글자가 문항 번호
        mean_val = merged.groupby('prob_num')['answerCode'].mean()
        merged['prob_num_mean'] = merged['prob_num'].map(mean_val)
        merged.drop('prob_num', axis=1, inplace=True)
        
        test_last_sequence['prob_num'] = test_last_sequence['assessmentItemID'].str[-3:]
        test_last_sequence['prob_num_mean'] = test_last_sequence['prob_num'].map(mean_val)
        test_last_sequence.drop('prob_num', axis=1, inplace=True)

        test_tmp['prob_num'] = test_tmp['assessmentItemID'].str[-3:]
        test_tmp['prob_num_mean'] = test_tmp['prob_num'].map(mean_val)
        test_tmp.drop('prob_num', axis=1, inplace=True)
        
        numeric_col.append('prob_num_mean')

        # 시험지의 각 문항 별 평균 정답률
        asses_mean = merged.groupby('assessmentItemID')['answerCode'].mean()
        merged['asses_mean'] = merged['assessmentItemID'].map(asses_mean)
        test_last_sequence['asses_mean'] = test_last_sequence['assessmentItemID'].map(asses_mean)
        test_tmp['asses_mean'] = test_tmp['assessmentItemID'].map(asses_mean)
        numeric_col.append('asses_mean')

        # 요일별 평균 정답률
        merged['days'] = merged['Timestamp'].dt.day_name()
        days_mean = merged.groupby('days')['answerCode'].mean()
        merged['days_mean'] = merged['days'].map(days_mean)
        merged.drop('days', axis=1, inplace=True)
        
        test_last_sequence['days'] = test_last_sequence['Timestamp'].dt.day_name()
        test_last_sequence['days_mean'] = test_last_sequence['days'].map(days_mean)
        test_last_sequence.drop('days', axis=1, inplace=True)

        test_tmp['days'] = test_tmp['Timestamp'].dt.day_name()
        test_tmp['days_mean'] = test_tmp['days'].map(days_mean)
        test_tmp.drop('days', axis=1, inplace=True)
        
        numeric_col.append('days_mean')

        # # FE04 에서 maxprob feature 추가하는 방법 참고
        # # 각 시험 속 문항번호를 수치형으로 만들어 추가한다.
        # merged['probnum'] = merged['assessmentItemID'].apply(lambda x: int(x[-3:]))
        # test_tmp['probnum'] = test_tmp['assessmentItemID'].apply(lambda x: int(x[-3:]))
        # test_last_sequence['probnum'] =test_last_sequence['assessmentItemID'].apply(lambda x: int(x[-3:]))

        # # 위 번호를 토대로 각 시험의 최종 문항을 피쳐로 추가한다.
        # merged_tmp = merged.groupby('testId')
        # merged_tmp = merged_tmp['probnum'].max()
        # merged['maxprob'] = merged['testId'].map(merged_tmp)

        # test_tmp['maxprob'] = test_tmp['testId'].map(merged_tmp)
        # test_last_sequence['maxprob'] = test_last_sequence['testId'].map(merged_tmp)
        # merged.drop(['probnum'], axis=1, inplace=True)
        # test_tmp.drop(['probnum'], axis=1, inplace=True)
        # test_last_sequence.drop(['probnum'], axis=1, inplace=True)

        ## 3. tag_groupby : KnowledgeTag 이용한 FE ## 
        tag_groupby = merged.groupby('KnowledgeTag').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        tag_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        tag_groupby['tag_grade'] = tag_groupby['meanAnswerRate'].apply(lambda x: grade_map(x)) # 태그 평균 정답률을 이용한 난이도 정의
        merged['tag_grade'] = merged['KnowledgeTag'].map(tag_groupby['tag_grade']) # merged mapping
        test_tmp['tag_grade'] = test_tmp['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_tmp mapping
        test_last_sequence['tag_grade'] = test_last_sequence['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_last_sequence mapping

        tag_solved_mean = tag_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 태그들은 몇 명에게 노출됐는가)
        tag_groupby['tag_solved'] = tag_groupby['numUsers'].apply(lambda x:int(x>tag_solved_mean)) # 태그가 많이 노출된 편인지, 아닌지 여부
        merged['tag_solved'] = merged['KnowledgeTag'].map(tag_groupby['tag_solved']) # merged mapping
        test_tmp['tag_solved'] = test_tmp['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_tmp mapping
        test_last_sequence['tag_solved'] = test_last_sequence['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_last_sequence mapping

        #### 4-1 : merged 에서 elapsed가 0인 문제들 대치 해주기
        # - 보통 시험의 마지막 문제는 elapsed가 0이다. (그 문제를 풀고 끝나기 때문에, 얼마나 걸렸는지 알 수가 없고, 그렇기 때문에 그 값을 0으로 대치하는 느낌)
        # - **이 값들을 효과적으로 대치할 수 있으면, test_last_sequence에 elapsed를 효과적으로 전달할 수 있기 때문에 미리 진행**
        # - 우선 merged에서, 시간이 900초 이상 (15분 이상)소요된 풀이시간은 모두 900초로 대치해주자 (대략 31421건)
        merged['elapsed'] = merged['elapsed'].apply(lambda x : 900 if x > 900 else x)

        # - elapsed가 0이 아닌것과 0인것을 나눠서 일단 쪼개고 (인덱스는 건들지 말자), elapsed가 0인 data frame에 유저별 문제풀이 시간의 중앙값으로 대치하고 다시 합쳐주자
        # - 합칠때는 concat으로 위아래로 붙인다음에 index 기준 정렬
        merged_elapsed_not0 = merged[merged.elapsed != 0]
        merged_elapsed_0 = merged[merged.elapsed == 0]
        merged_elapsed_0['elapsed'] = merged_elapsed_0['userID'].map(merged.groupby('userID')['elapsed'].median())

        merged = pd.concat([merged_elapsed_not0,merged_elapsed_0], axis=0)
        merged = merged.sort_index()
        
        # - 이제 test_last_sequence에 있는 elapsed가 0인 애들은, 다른 사람들은 그 문제를 푸는데 얼마나 걸렸는지를 기준으로 대치할 수 있게 되었다
        test_last_sequence['elapsed'] = test_last_sequence['assessmentItemID'].map(merged.groupby('assessmentItemID')['elapsed'].median())

        ########### ----------------------------------- ###########
        ########### test_tmp랑 test_sequence랑 합쳐준다 ###########
        # - 그러고 합치자 (test_tmp랑 test_last_sequence랑)
        test_df = pd.concat([test_tmp, test_last_sequence], axis=0).sort_index()

        # - 이제 elapsed가 잘 대치 되어있기 때문에, mark_randomly feature를 만들 수 있다.
        merged['mark_randomly'] = merged['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주
        test_df['mark_randomly'] = test_df['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주

        # 수치형 feature 정규화
        scaler = StandardScaler()
        scaler.fit(merged[numeric_col])
        merged[numeric_col] = scaler.transform(merged[numeric_col])
        test_df[numeric_col] = scaler.transform(test_df[numeric_col])
        
        # data leakage 허용 : merged가 train data 의 역할을 하자
        train_df = merged
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


class FE08(FeatureEngineer):
    def __str__(self):
        return \
            """FE06 + FE04"""
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
        
        numeric_col = []

        #### 1. train_df, test_df 에서 interaction, elapsed 구해놓기 ####
        # train_df = pd.read_csv('../data/train_data.csv')
        # test_df = pd.read_csv('../data/test_data.csv')
        train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode'].fillna(-1)
        test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode'].fillna(-1)

        diff = train_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        train_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간

        diff = test_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        test_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간        
        
        #### 2. test_df 에서 test_tmp, test_last_sequence 떼어내기 ####
        # - test_tmp : not -1
        # - test_last_sequence : only -1
        # merge를 하기 위해서 test_df에서 -1을 answerCode로 갖는 행을 제외한 df 생성
        test_tmp = test_df[test_df.answerCode != -1]
        test_last_sequence = test_df[test_df.answerCode == -1]

        #### 3. train_df + test_tmp = merged 로 concat하기 ####
        # merge후, interaction항 추가 해줌
        merged = pd.concat([train_df,test_tmp],axis=0)
        merged.sort_values(['userID','Timestamp'], inplace=True)
        merged = merged.reset_index(drop=True) #.drop('index', axis=1)
        
        #### 4. merged 기준으로 FE를 진행, test_tmp와 test_last_sequence에도 각각의 정보(userID, assessmentItemID)를 이용해서 mapping ####
        ## 1. stu_groupby_merged : counts, user_grade 추가 ##
        stu_groupby_merged = merged.groupby('userID').agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        })
        stu_groupby_merged.columns = ['counts', 'meanAnswerRate'] # groupby 집계, counts : 유저가 푼 문제의 개수
        merged['counts'] = merged['userID'].map(stu_groupby_merged['counts']) # merged mapping
        test_tmp['counts'] = test_tmp['userID'].map(stu_groupby_merged['counts']) # test_tmp mapping
        test_last_sequence['counts'] = test_last_sequence['userID'].map(stu_groupby_merged['counts']) # test_last_sequence mapping

        stu_groupby_merged['user_grade'] = stu_groupby_merged['meanAnswerRate'].apply(grade_map) # 유저의 평균 정답률을 이용한 실력,등급 정의
        merged['user_grade'] = merged['userID'].map(stu_groupby_merged['user_grade']) # merged mapping
        test_tmp['user_grade'] = test_tmp['userID'].map(stu_groupby_merged['user_grade']) # test_tmp mapping
        test_last_sequence['user_grade'] = test_last_sequence['userID'].map(stu_groupby_merged['user_grade']) # test_last_sequence mapping

        ## 2. prob_groupby : assessmentItemID 이용한 FE ## 
        prob_groupby = merged.groupby('assessmentItemID').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        prob_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        prob_groupby['ass_grade'] = prob_groupby['meanAnswerRate'].apply(grade_map) # 문제 평균 정답률을 이용한 난이도 정의
        merged['ass_grade'] = merged['assessmentItemID'].map(prob_groupby['ass_grade']) # merged mapping
        test_tmp['ass_grade'] = test_tmp['assessmentItemID'].map(prob_groupby['ass_grade']) # test_tmp mapping
        test_last_sequence['ass_grade'] = test_last_sequence['assessmentItemID'].map(prob_groupby['ass_grade']) # test_last_sequence mapping

        prob_solved_mean = prob_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 문제들은 몇 명에게 노출됐는가)
        prob_groupby['ass_solved'] = prob_groupby['numUsers'].apply(lambda x: int(x>prob_solved_mean)) # 문제가 많이 노출된 편인지, 아닌지 여부
        merged['ass_solved'] = merged['assessmentItemID'].map(prob_groupby['ass_solved']) # merged mapping
        test_tmp['ass_solved'] = test_tmp['assessmentItemID'].map(prob_groupby['ass_solved']) # test_tmp mapping
        test_last_sequence['ass_solved'] = test_last_sequence['assessmentItemID'].map(prob_groupby['ass_solved']) # test_last_sequence mapping 

        # FE05 내용 추가
        # 시험지 문항 번호별 평균 정답률
        merged['prob_num'] = merged['assessmentItemID'].str[-3:] # assessmentItemID의 마지막 3글자가 문항 번호
        mean_val = merged.groupby('prob_num')['answerCode'].mean()
        merged['prob_num_mean'] = merged['prob_num'].map(mean_val)
        merged.drop('prob_num', axis=1, inplace=True)
        
        test_last_sequence['prob_num'] = test_last_sequence['assessmentItemID'].str[-3:]
        test_last_sequence['prob_num_mean'] = test_last_sequence['prob_num'].map(mean_val)
        test_last_sequence.drop('prob_num', axis=1, inplace=True)

        test_tmp['prob_num'] = test_tmp['assessmentItemID'].str[-3:]
        test_tmp['prob_num_mean'] = test_tmp['prob_num'].map(mean_val)
        test_tmp.drop('prob_num', axis=1, inplace=True)
        
        numeric_col.append('prob_num_mean')

        # 시험지의 각 문항 별 평균 정답률
        asses_mean = merged.groupby('assessmentItemID')['answerCode'].mean()
        merged['asses_mean'] = merged['assessmentItemID'].map(asses_mean)
        test_last_sequence['asses_mean'] = test_last_sequence['assessmentItemID'].map(asses_mean)
        test_tmp['asses_mean'] = test_tmp['assessmentItemID'].map(asses_mean)
        numeric_col.append('asses_mean')

        # 요일별 평균 정답률
        merged['days'] = merged['Timestamp'].dt.day_name()
        days_mean = merged.groupby('days')['answerCode'].mean()
        merged['days_mean'] = merged['days'].map(days_mean)
        merged.drop('days', axis=1, inplace=True)
        
        test_last_sequence['days'] = test_last_sequence['Timestamp'].dt.day_name()
        test_last_sequence['days_mean'] = test_last_sequence['days'].map(days_mean)
        test_last_sequence.drop('days', axis=1, inplace=True)

        test_tmp['days'] = test_tmp['Timestamp'].dt.day_name()
        test_tmp['days_mean'] = test_tmp['days'].map(days_mean)
        test_tmp.drop('days', axis=1, inplace=True)
        
        numeric_col.append('days_mean')

        # FE04 에서 maxprob feature 추가하는 방법 참고
        # 각 시험 속 문항번호를 수치형으로 만들어 추가한다.
        merged['probnum'] = merged['assessmentItemID'].apply(lambda x: int(x[-3:]))
        test_tmp['probnum'] = test_tmp['assessmentItemID'].apply(lambda x: int(x[-3:]))
        test_last_sequence['probnum'] =test_last_sequence['assessmentItemID'].apply(lambda x: int(x[-3:]))

        # 위 번호를 토대로 각 시험의 최종 문항을 피쳐로 추가한다.
        merged_tmp = merged.groupby('testId')
        merged_tmp = merged_tmp['probnum'].max()
        merged['maxprob'] = merged['testId'].map(merged_tmp)

        test_tmp['maxprob'] = test_tmp['testId'].map(merged_tmp)
        test_last_sequence['maxprob'] = test_last_sequence['testId'].map(merged_tmp)
        merged.drop(['probnum'], axis=1, inplace=True)
        test_tmp.drop(['probnum'], axis=1, inplace=True)
        test_last_sequence.drop(['probnum'], axis=1, inplace=True)

        ## 3. tag_groupby : KnowledgeTag 이용한 FE ## 
        tag_groupby = merged.groupby('KnowledgeTag').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        tag_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        tag_groupby['tag_grade'] = tag_groupby['meanAnswerRate'].apply(lambda x: grade_map(x)) # 태그 평균 정답률을 이용한 난이도 정의
        merged['tag_grade'] = merged['KnowledgeTag'].map(tag_groupby['tag_grade']) # merged mapping
        test_tmp['tag_grade'] = test_tmp['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_tmp mapping
        test_last_sequence['tag_grade'] = test_last_sequence['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_last_sequence mapping

        tag_solved_mean = tag_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 태그들은 몇 명에게 노출됐는가)
        tag_groupby['tag_solved'] = tag_groupby['numUsers'].apply(lambda x:int(x>tag_solved_mean)) # 태그가 많이 노출된 편인지, 아닌지 여부
        merged['tag_solved'] = merged['KnowledgeTag'].map(tag_groupby['tag_solved']) # merged mapping
        test_tmp['tag_solved'] = test_tmp['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_tmp mapping
        test_last_sequence['tag_solved'] = test_last_sequence['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_last_sequence mapping

        #### 4-1 : merged 에서 elapsed가 0인 문제들 대치 해주기
        # - 보통 시험의 마지막 문제는 elapsed가 0이다. (그 문제를 풀고 끝나기 때문에, 얼마나 걸렸는지 알 수가 없고, 그렇기 때문에 그 값을 0으로 대치하는 느낌)
        # - **이 값들을 효과적으로 대치할 수 있으면, test_last_sequence에 elapsed를 효과적으로 전달할 수 있기 때문에 미리 진행**
        # - 우선 merged에서, 시간이 900초 이상 (15분 이상)소요된 풀이시간은 모두 900초로 대치해주자 (대략 31421건)
        merged['elapsed'] = merged['elapsed'].apply(lambda x : 900 if x > 900 else x)

        # - elapsed가 0이 아닌것과 0인것을 나눠서 일단 쪼개고 (인덱스는 건들지 말자), elapsed가 0인 data frame에 유저별 문제풀이 시간의 중앙값으로 대치하고 다시 합쳐주자
        # - 합칠때는 concat으로 위아래로 붙인다음에 index 기준 정렬
        merged_elapsed_not0 = merged[merged.elapsed != 0]
        merged_elapsed_0 = merged[merged.elapsed == 0]
        merged_elapsed_0['elapsed'] = merged_elapsed_0['userID'].map(merged.groupby('userID')['elapsed'].median())

        merged = pd.concat([merged_elapsed_not0,merged_elapsed_0], axis=0)
        merged = merged.sort_index()
        
        # - 이제 test_last_sequence에 있는 elapsed가 0인 애들은, 다른 사람들은 그 문제를 푸는데 얼마나 걸렸는지를 기준으로 대치할 수 있게 되었다
        test_last_sequence['elapsed'] = test_last_sequence['assessmentItemID'].map(merged.groupby('assessmentItemID')['elapsed'].median())

        ########### ----------------------------------- ###########
        ########### test_tmp랑 test_sequence랑 합쳐준다 ###########
        # - 그러고 합치자 (test_tmp랑 test_last_sequence랑)
        test_df = pd.concat([test_tmp, test_last_sequence], axis=0).sort_index()

        # - 이제 elapsed가 잘 대치 되어있기 때문에, mark_randomly feature를 만들 수 있다.
        merged['mark_randomly'] = merged['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주
        test_df['mark_randomly'] = test_df['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주

        # 수치형 feature 정규화
        scaler = StandardScaler()
        scaler.fit(merged[numeric_col])
        merged[numeric_col] = scaler.transform(merged[numeric_col])
        test_df[numeric_col] = scaler.transform(test_df[numeric_col])
        
        # data leakage 허용 : merged가 train data 의 역할을 하자
        train_df = merged
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


class FE09(FeatureEngineer):
    def __str__(self):
        return \
            """FE08 + add interaciton 1-5 terms"""
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
        
        numeric_col = []

        #### 1. train_df, test_df 에서 interaction, elapsed 구해놓기 ####
        # train_df = pd.read_csv('../data/train_data.csv')
        # test_df = pd.read_csv('../data/test_data.csv')
        train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'], format="%Y-%m-%d %H:%M:%S")
        train_df['interaction'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode'].fillna(-1).astype(np.int16)
        test_df['interaction'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode'].fillna(-1).astype(np.int16)
        # interaction2
        train_df['interaction_2'] = train_df.groupby(['userID','testId'])[['answerCode']].shift(2)['answerCode'].fillna(-1).astype(np.int16)
        test_df['interaction_2'] = test_df.groupby(['userID','testId'])[['answerCode']].shift(2)['answerCode'].fillna(-1).astype(np.int16)
        # interaction3
        train_df['interaction_3'] = train_df.groupby(['userID','testId'])[['answerCode']].shift(3)['answerCode'].fillna(-1).astype(np.int16)
        test_df['interaction_3'] = test_df.groupby(['userID','testId'])[['answerCode']].shift(3)['answerCode'].fillna(-1).astype(np.int16)
        # interaction4
        train_df['interaction_4'] = train_df.groupby(['userID','testId'])[['answerCode']].shift(4)['answerCode'].fillna(-1).astype(np.int16)
        test_df['interaction_4'] = test_df.groupby(['userID','testId'])[['answerCode']].shift(4)['answerCode'].fillna(-1).astype(np.int16)
        # interaction5
        train_df['interaction_5'] = train_df.groupby(['userID','testId'])[['answerCode']].shift(5)['answerCode'].fillna(-1).astype(np.int16)
        test_df['interaction_5'] = test_df.groupby(['userID','testId'])[['answerCode']].shift(5)['answerCode'].fillna(-1).astype(np.int16)


        diff = train_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        train_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간

        diff = test_df.loc[:, ['userID','testId','Timestamp']].groupby(['userID','testId']).diff().fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())
        test_df['elapsed'] = pd.concat([diff[1:], pd.Series([0.0])]).reset_index().iloc[:,1]  # 걸린 시간        
        
        #### 2. test_df 에서 test_tmp, test_last_sequence 떼어내기 ####
        # - test_tmp : not -1
        # - test_last_sequence : only -1
        # merge를 하기 위해서 test_df에서 -1을 answerCode로 갖는 행을 제외한 df 생성
        test_tmp = test_df[test_df.answerCode != -1]
        test_last_sequence = test_df[test_df.answerCode == -1]

        #### 3. train_df + test_tmp = merged 로 concat하기 ####
        # merge후, interaction항 추가 해줌
        merged = pd.concat([train_df,test_tmp],axis=0)
        merged.sort_values(['userID','Timestamp'], inplace=True)
        merged = merged.reset_index(drop=True) #.drop('index', axis=1)
        
        #### 4. merged 기준으로 FE를 진행, test_tmp와 test_last_sequence에도 각각의 정보(userID, assessmentItemID)를 이용해서 mapping ####
        ## 1. stu_groupby_merged : counts, user_grade 추가 ##
        stu_groupby_merged = merged.groupby('userID').agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        })
        stu_groupby_merged.columns = ['counts', 'meanAnswerRate'] # groupby 집계, counts : 유저가 푼 문제의 개수
        merged['counts'] = merged['userID'].map(stu_groupby_merged['counts']) # merged mapping
        test_tmp['counts'] = test_tmp['userID'].map(stu_groupby_merged['counts']) # test_tmp mapping
        test_last_sequence['counts'] = test_last_sequence['userID'].map(stu_groupby_merged['counts']) # test_last_sequence mapping

        stu_groupby_merged['user_grade'] = stu_groupby_merged['meanAnswerRate'].apply(grade_map) # 유저의 평균 정답률을 이용한 실력,등급 정의
        merged['user_grade'] = merged['userID'].map(stu_groupby_merged['user_grade']) # merged mapping
        test_tmp['user_grade'] = test_tmp['userID'].map(stu_groupby_merged['user_grade']) # test_tmp mapping
        test_last_sequence['user_grade'] = test_last_sequence['userID'].map(stu_groupby_merged['user_grade']) # test_last_sequence mapping

        ## 2. prob_groupby : assessmentItemID 이용한 FE ## 
        prob_groupby = merged.groupby('assessmentItemID').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        prob_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        prob_groupby['ass_grade'] = prob_groupby['meanAnswerRate'].apply(grade_map) # 문제 평균 정답률을 이용한 난이도 정의
        merged['ass_grade'] = merged['assessmentItemID'].map(prob_groupby['ass_grade']) # merged mapping
        test_tmp['ass_grade'] = test_tmp['assessmentItemID'].map(prob_groupby['ass_grade']) # test_tmp mapping
        test_last_sequence['ass_grade'] = test_last_sequence['assessmentItemID'].map(prob_groupby['ass_grade']) # test_last_sequence mapping

        prob_solved_mean = prob_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 문제들은 몇 명에게 노출됐는가)
        prob_groupby['ass_solved'] = prob_groupby['numUsers'].apply(lambda x: int(x>prob_solved_mean)) # 문제가 많이 노출된 편인지, 아닌지 여부
        merged['ass_solved'] = merged['assessmentItemID'].map(prob_groupby['ass_solved']) # merged mapping
        test_tmp['ass_solved'] = test_tmp['assessmentItemID'].map(prob_groupby['ass_solved']) # test_tmp mapping
        test_last_sequence['ass_solved'] = test_last_sequence['assessmentItemID'].map(prob_groupby['ass_solved']) # test_last_sequence mapping 

        # FE05 내용 추가
        # 시험지 문항 번호별 평균 정답률
        merged['prob_num'] = merged['assessmentItemID'].str[-3:] # assessmentItemID의 마지막 3글자가 문항 번호
        mean_val = merged.groupby('prob_num')['answerCode'].mean()
        merged['prob_num_mean'] = merged['prob_num'].map(mean_val)
        merged.drop('prob_num', axis=1, inplace=True)
        
        test_last_sequence['prob_num'] = test_last_sequence['assessmentItemID'].str[-3:]
        test_last_sequence['prob_num_mean'] = test_last_sequence['prob_num'].map(mean_val)
        test_last_sequence.drop('prob_num', axis=1, inplace=True)

        test_tmp['prob_num'] = test_tmp['assessmentItemID'].str[-3:]
        test_tmp['prob_num_mean'] = test_tmp['prob_num'].map(mean_val)
        test_tmp.drop('prob_num', axis=1, inplace=True)
        
        numeric_col.append('prob_num_mean')

        # 시험지의 각 문항 별 평균 정답률
        asses_mean = merged.groupby('assessmentItemID')['answerCode'].mean()
        merged['asses_mean'] = merged['assessmentItemID'].map(asses_mean)
        test_last_sequence['asses_mean'] = test_last_sequence['assessmentItemID'].map(asses_mean)
        test_tmp['asses_mean'] = test_tmp['assessmentItemID'].map(asses_mean)
        numeric_col.append('asses_mean')

        # 요일별 평균 정답률
        merged['days'] = merged['Timestamp'].dt.day_name()
        days_mean = merged.groupby('days')['answerCode'].mean()
        merged['days_mean'] = merged['days'].map(days_mean)
        merged.drop('days', axis=1, inplace=True)
        
        test_last_sequence['days'] = test_last_sequence['Timestamp'].dt.day_name()
        test_last_sequence['days_mean'] = test_last_sequence['days'].map(days_mean)
        test_last_sequence.drop('days', axis=1, inplace=True)

        test_tmp['days'] = test_tmp['Timestamp'].dt.day_name()
        test_tmp['days_mean'] = test_tmp['days'].map(days_mean)
        test_tmp.drop('days', axis=1, inplace=True)
        
        numeric_col.append('days_mean')

        # FE04 에서 maxprob feature 추가하는 방법 참고
        # 각 시험 속 문항번호를 수치형으로 만들어 추가한다.
        merged['probnum'] = merged['assessmentItemID'].apply(lambda x: int(x[-3:]))
        test_tmp['probnum'] = test_tmp['assessmentItemID'].apply(lambda x: int(x[-3:]))
        test_last_sequence['probnum'] =test_last_sequence['assessmentItemID'].apply(lambda x: int(x[-3:]))

        # 위 번호를 토대로 각 시험의 최종 문항을 피쳐로 추가한다.
        merged_tmp = merged.groupby('testId')
        merged_tmp = merged_tmp['probnum'].max()
        merged['maxprob'] = merged['testId'].map(merged_tmp)

        test_tmp['maxprob'] = test_tmp['testId'].map(merged_tmp)
        test_last_sequence['maxprob'] = test_last_sequence['testId'].map(merged_tmp)
        merged.drop(['probnum'], axis=1, inplace=True)
        test_tmp.drop(['probnum'], axis=1, inplace=True)
        test_last_sequence.drop(['probnum'], axis=1, inplace=True)

        ## 3. tag_groupby : KnowledgeTag 이용한 FE ## 
        tag_groupby = merged.groupby('KnowledgeTag').agg({
            'userID': 'count',
            'answerCode': percentile
        })
        tag_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
        tag_groupby['tag_grade'] = tag_groupby['meanAnswerRate'].apply(lambda x: grade_map(x)) # 태그 평균 정답률을 이용한 난이도 정의
        merged['tag_grade'] = merged['KnowledgeTag'].map(tag_groupby['tag_grade']) # merged mapping
        test_tmp['tag_grade'] = test_tmp['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_tmp mapping
        test_last_sequence['tag_grade'] = test_last_sequence['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_last_sequence mapping

        tag_solved_mean = tag_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 태그들은 몇 명에게 노출됐는가)
        tag_groupby['tag_solved'] = tag_groupby['numUsers'].apply(lambda x:int(x>tag_solved_mean)) # 태그가 많이 노출된 편인지, 아닌지 여부
        merged['tag_solved'] = merged['KnowledgeTag'].map(tag_groupby['tag_solved']) # merged mapping
        test_tmp['tag_solved'] = test_tmp['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_tmp mapping
        test_last_sequence['tag_solved'] = test_last_sequence['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_last_sequence mapping

        #### 4-1 : merged 에서 elapsed가 0인 문제들 대치 해주기
        # - 보통 시험의 마지막 문제는 elapsed가 0이다. (그 문제를 풀고 끝나기 때문에, 얼마나 걸렸는지 알 수가 없고, 그렇기 때문에 그 값을 0으로 대치하는 느낌)
        # - **이 값들을 효과적으로 대치할 수 있으면, test_last_sequence에 elapsed를 효과적으로 전달할 수 있기 때문에 미리 진행**
        # - 우선 merged에서, 시간이 900초 이상 (15분 이상)소요된 풀이시간은 모두 900초로 대치해주자 (대략 31421건)
        merged['elapsed'] = merged['elapsed'].apply(lambda x : 900 if x > 900 else x)

        # - elapsed가 0이 아닌것과 0인것을 나눠서 일단 쪼개고 (인덱스는 건들지 말자), elapsed가 0인 data frame에 유저별 문제풀이 시간의 중앙값으로 대치하고 다시 합쳐주자
        # - 합칠때는 concat으로 위아래로 붙인다음에 index 기준 정렬
        merged_elapsed_not0 = merged[merged.elapsed != 0]
        merged_elapsed_0 = merged[merged.elapsed == 0]
        merged_elapsed_0['elapsed'] = merged_elapsed_0['userID'].map(merged.groupby('userID')['elapsed'].median())

        merged = pd.concat([merged_elapsed_not0,merged_elapsed_0], axis=0)
        merged = merged.sort_index()
        
        # - 이제 test_last_sequence에 있는 elapsed가 0인 애들은, 다른 사람들은 그 문제를 푸는데 얼마나 걸렸는지를 기준으로 대치할 수 있게 되었다
        test_last_sequence['elapsed'] = test_last_sequence['assessmentItemID'].map(merged.groupby('assessmentItemID')['elapsed'].median())

        ########### ----------------------------------- ###########
        ########### test_tmp랑 test_sequence랑 합쳐준다 ###########
        # - 그러고 합치자 (test_tmp랑 test_last_sequence랑)
        test_df = pd.concat([test_tmp, test_last_sequence], axis=0).sort_index()

        # - 이제 elapsed가 잘 대치 되어있기 때문에, mark_randomly feature를 만들 수 있다.
        merged['mark_randomly'] = merged['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주
        test_df['mark_randomly'] = test_df['elapsed'].apply(lambda x: int((x>0) & (x<=5)))     # 걸린 시간이 1초에서 5초 사이는 평균 정답률이 너무 낮아서 찍은 걸로 간주

        # 수치형 feature 정규화
        scaler = StandardScaler()
        scaler.fit(merged[numeric_col])
        merged[numeric_col] = scaler.transform(merged[numeric_col])
        test_df[numeric_col] = scaler.transform(test_df[numeric_col])
        
        # data leakage 허용 : merged가 train data 의 역할을 하자
        train_df = merged
        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction' : 'interaction_c',
                'interaction_2' : 'interaction_2_c',
                'interaction_3' : 'interaction_3_c',
                'interaction_4' : 'interaction_4_c',
                'interaction_5' : 'interaction_5_c',
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
                'interaction_2' : 'interaction_2_c',
                'interaction_3' : 'interaction_3_c',
                'interaction_4' : 'interaction_4_c',
                'interaction_5' : 'interaction_5_c',
                'user_grade' : 'user_grade_c',
                'ass_grade' : 'ass_grade_c',
                'ass_solved' : 'ass_solved_c',
                'tag_grade' : 'tag_grade_c',
                'tag_solved' : 'tag_solved_c',
                'mark_randomly' : 'mark_randomly_c'
            }
        )
        return train_df, test_df


class FE10(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3 만 추가"""
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        # train_df['cont_ex'] = 0.0
        # test_df['cont_ex'] = 0.0

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
            }
        )
        return train_df, test_df


class FE11(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2 만 추가"""
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        # train_df['cont_ex'] = 0.0
        # test_df['cont_ex'] = 0.0

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
            }
        )
        return train_df, test_df

class FE12(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 만 추가"""
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']

        # train_df['cont_ex'] = 0.0
        # test_df['cont_ex'] = 0.0

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
            }
        )
        return train_df, test_df

class FE15(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 만 추가"""
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']


        # train_df['cont_ex'] = 0.0
        # test_df['cont_ex'] = 0.0

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
            }
        )
        return train_df, test_df


class FE16(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 추가
            testId 대분류 추가(난이도 판별됨)
            """
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']



        train_df['testId_large'] = train_df['testId'].apply(lambda s: s[2])
        test_df['testId_large'] = test_df['testId'].apply(lambda s: s[2])
        
      
        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
            }
        )
        return train_df, test_df


class FE17(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 추가
            testId 대분류 추가(난이도 판별됨)
            assessment 별 정답률 카테고라이즈 추가 (문항별 난이도느낌)
            """
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']



        train_df['testId_large'] = train_df['testId'].apply(lambda s: s[2])
        test_df['testId_large'] = test_df['testId'].apply(lambda s: s[2])

        def categorize(x):
            if  x <= 0.1:
                return 0
            elif 0.1 < x <= 0.2:
                return 1
            elif 0.2 < x <= 0.3:
                return 2
            elif 0.3 < x <= 0.4:
                return 3
            elif 0.4 < x <= 0.5:
                return 4
            elif 0.5 < x <= 0.6:
                return 5
            elif 0.6 < x <= 0.7:
                return 6
            elif 0.7 < x <= 0.8:
                return 7
            elif 0.8 < x <= 0.9:
                return 8
            else:
                return 9


        train_assessment_rates = train_df.groupby('assessmentItemID')['answerCode'].mean()

        train_df['assessment_rate'] = train_df['assessmentItemID'].map(train_assessment_rates.map(categorize))
        test_df['assessment_rate'] = test_df['assessmentItemID'].map(train_assessment_rates.map(categorize))


        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                # 'user_rate' : 'user_rate_c',
                # # 'train_know_rates': 'train_know_rates_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                # 'user_rate' : 'user_rate_c',
                # # 'train_know_rates': 'train_know_rates_c',
            }
        )
        return train_df, test_df


class FE18(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 추가
            testId 대분류 추가(난이도 판별됨)
            assessment 별 정답률 카테고라이즈 추가 (문항별 난이도느낌)
            user 별 푼 문제들에 대한 평균 정답률 추가
            """
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']



        train_df['testId_large'] = train_df['testId'].apply(lambda s: s[2])
        test_df['testId_large'] = test_df['testId'].apply(lambda s: s[2])

        
        def categorize(x):
            if  x <= 0.1:
                return 0
            elif 0.1 < x <= 0.2:
                return 1
            elif 0.2 < x <= 0.3:
                return 2
            elif 0.3 < x <= 0.4:
                return 3
            elif 0.4 < x <= 0.5:
                return 4
            elif 0.5 < x <= 0.6:
                return 5
            elif 0.6 < x <= 0.7:
                return 6
            elif 0.7 < x <= 0.8:
                return 7
            elif 0.8 < x <= 0.9:
                return 8
            else:
                return 9


        train_assessment_rates = train_df.groupby('assessmentItemID')['answerCode'].mean()

        train_df['assessment_rate'] = train_df['assessmentItemID'].map(train_assessment_rates.map(categorize))
        test_df['assessment_rate'] = test_df['assessmentItemID'].map(train_assessment_rates.map(categorize))



        train_user_rate = train_df.groupby('userID')['answerCode'].mean()
        test_user_rate = test_df.groupby('userID')['answerCode'].mean()

        train_df['user_rate'] = train_df['userID'].map(train_user_rate.map(categorize))
        test_df['user_rate'] = test_df['userID'].map(test_user_rate.map(categorize))
        

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                # 'train_know_rates': 'train_know_rates_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                # 'train_know_rates': 'train_know_rates_c',
            }
        )
        return train_df, test_df


class FE19(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 추가
            testId 대분류 추가(난이도 판별됨)
            assessment 별 정답률 카테고라이즈 추가 (문항별 난이도느낌)
            user 별 푼 문제들에 대한 평균 정답률 추가
            유저, 시험별 타임 일랩스 추가
            """
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']



        train_df['testId_large'] = train_df['testId'].apply(lambda s: s[2])
        test_df['testId_large'] = test_df['testId'].apply(lambda s: s[2])

        def categorize(x):
            if  x <= 0.1:
                return 0
            elif 0.1 < x <= 0.2:
                return 1
            elif 0.2 < x <= 0.3:
                return 2
            elif 0.3 < x <= 0.4:
                return 3
            elif 0.4 < x <= 0.5:
                return 4
            elif 0.5 < x <= 0.6:
                return 5
            elif 0.6 < x <= 0.7:
                return 6
            elif 0.7 < x <= 0.8:
                return 7
            elif 0.8 < x <= 0.9:
                return 8
            else:
                return 9


        train_assessment_rates = train_df.groupby('assessmentItemID')['answerCode'].mean()

        train_df['assessment_rate'] = train_df['assessmentItemID'].map(train_assessment_rates.map(categorize))
        test_df['assessment_rate'] = test_df['assessmentItemID'].map(train_assessment_rates.map(categorize))



        train_user_rate = train_df.groupby('userID')['answerCode'].mean()
        test_user_rate = test_df.groupby('userID')['answerCode'].mean()

        train_df['user_rate'] = train_df['userID'].map(train_user_rate.map(categorize))
        test_df['user_rate'] = test_df['userID'].map(test_user_rate.map(categorize))
        


        
        train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
        train_df['elapse'] = train_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
        train_df.loc[train_df['testId'] != train_df['testId'].shift(-1), 'elapse'] = pd.NaT
        train_df['elapse'] = train_df['elapse'].dt.seconds
        train_df.loc[(train_df['elapse'] > 600) | (train_df['elapse'] < 2), 'elapse'] = np.nan
        train_df['elapse'].fillna(0)

        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
        test_df['elapse'] = test_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
        test_df.loc[test_df['testId'] != test_df['testId'].shift(-1), 'elapse'] = pd.NaT
        test_df['elapse'] = test_df['elapse'].dt.seconds
        test_df.loc[(test_df['elapse'] > 600) | (test_df['elapse'] < 2), 'elapse'] = np.nan
        test_df['elapse'].fillna(0)


        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                # 'train_know_rates': 'train_know_rates_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                # 'train_know_rates': 'train_know_rates_c',
            }
        )
        return train_df, test_df


class FE20(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 추가
            testId 대분류 추가(난이도 판별됨)
            assessment 별 정답률 카테고라이즈 추가 (문항별 난이도느낌)
            user 별 푼 문제들에 대한 평균 정답률 추가
            유저, 시험별 타임 일랩스 추가
            지식태그별 정답률 추가
            """
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']



        train_df['testId_large'] = train_df['testId'].apply(lambda s: s[2])
        test_df['testId_large'] = test_df['testId'].apply(lambda s: s[2])

        def categorize(x):
            if  x <= 0.1:
                return 0
            elif 0.1 < x <= 0.2:
                return 1
            elif 0.2 < x <= 0.3:
                return 2
            elif 0.3 < x <= 0.4:
                return 3
            elif 0.4 < x <= 0.5:
                return 4
            elif 0.5 < x <= 0.6:
                return 5
            elif 0.6 < x <= 0.7:
                return 6
            elif 0.7 < x <= 0.8:
                return 7
            elif 0.8 < x <= 0.9:
                return 8
            else:
                return 9


        train_assessment_rates = train_df.groupby('assessmentItemID')['answerCode'].mean()

        train_df['assessment_rate'] = train_df['assessmentItemID'].map(train_assessment_rates.map(categorize))
        test_df['assessment_rate'] = test_df['assessmentItemID'].map(train_assessment_rates.map(categorize))



        train_user_rate = train_df.groupby('userID')['answerCode'].mean()
        test_user_rate = test_df.groupby('userID')['answerCode'].mean()

        train_df['user_rate'] = train_df['userID'].map(train_user_rate.map(categorize))
        test_df['user_rate'] = test_df['userID'].map(test_user_rate.map(categorize))
        
        


        train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
        train_df['elapse'] = train_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
        train_df.loc[train_df['testId'] != train_df['testId'].shift(-1), 'elapse'] = pd.NaT
        train_df['elapse'] = train_df['elapse'].dt.seconds
        train_df.loc[(train_df['elapse'] > 600) | (train_df['elapse'] < 2), 'elapse'] = np.nan
        train_df['elapse'].fillna(0)

        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
        test_df['elapse'] = test_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
        test_df.loc[test_df['testId'] != test_df['testId'].shift(-1), 'elapse'] = pd.NaT
        test_df['elapse'] = test_df['elapse'].dt.seconds
        test_df.loc[(test_df['elapse'] > 600) | (test_df['elapse'] < 2), 'elapse'] = np.nan
        test_df['elapse'].fillna(0)

        # train_df['window5'] = train_df.groupby('userID')['answerCode'].rolling(window=5).mean().reset_index()['answerCode']
        # test_df['window5'] = test_df.groupby('userID')['answerCode'].rolling(window=5).mean().reset_index()['answerCode']



        train_know_rates = train_df.groupby('KnowledgeTag')['answerCode'].mean()

        train_df['know_rates'] = train_df['KnowledgeTag'].map(train_know_rates.map(categorize))
        test_df['know_rates'] = test_df['KnowledgeTag'].map(train_know_rates.map(categorize))

        # 카테고리 컬럼 끝 _c 붙여주세요.
        train_df = train_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                'train_know_rates': 'train_know_rates_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                'train_know_rates': 'train_know_rates_c',
            }
        )
        return train_df, test_df
    


class FE21(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 추가
            testId 대분류 추가(난이도 판별됨)
            assessment 별 정답률 카테고라이즈 추가 (문항별 난이도느낌)
            user 별 푼 문제들에 대한 평균 정답률 추가
            유저, 시험별 타임 일랩스 추가
            모두 머지 적용
            """
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']

        merged_df = pd.concat([train_df, test_df[test_df['answerCode'] != -1]])

        merged_df['testId_large'] = merged_df['testId'].apply(lambda s: s[2])
        test_df['testId_large'] = test_df['testId'].apply(lambda s: s[2])

        def categorize(x):
            if  x <= 0.1:
                return 0
            elif 0.1 < x <= 0.2:
                return 1
            elif 0.2 < x <= 0.3:
                return 2
            elif 0.3 < x <= 0.4:
                return 3
            elif 0.4 < x <= 0.5:
                return 4
            elif 0.5 < x <= 0.6:
                return 5
            elif 0.6 < x <= 0.7:
                return 6
            elif 0.7 < x <= 0.8:
                return 7
            elif 0.8 < x <= 0.9:
                return 8
            else:
                return 9


        merged_assessment_rates = merged_df.groupby('assessmentItemID')['answerCode'].mean()

        merged_df['assessment_rate'] = merged_df['assessmentItemID'].map(merged_assessment_rates.map(categorize))
        test_df['assessment_rate'] = test_df['assessmentItemID'].map(merged_assessment_rates.map(categorize))


        merged_user_rate = merged_df.groupby('userID')['answerCode'].mean()
        test_user_rate = test_df.groupby('userID')['answerCode'].mean()

        merged_df['user_rate'] = merged_df['userID'].map(merged_user_rate.map(categorize))
        test_df['user_rate'] = test_df['userID'].map(test_user_rate.map(categorize))
        


        merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])
        merged_df['elapse'] = merged_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
        merged_df.loc[merged_df['testId'] != merged_df['testId'].shift(-1), 'elapse'] = pd.NaT
        merged_df['elapse'] = merged_df['elapse'].dt.seconds
        merged_df.loc[(merged_df['elapse'] > 600) | (merged_df['elapse'] < 2), 'elapse'] = np.nan
        merged_df['elapse'].fillna(0)

        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
        test_df['elapse'] = test_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
        test_df.loc[test_df['testId'] != test_df['testId'].shift(-1), 'elapse'] = pd.NaT
        test_df['elapse'] = test_df['elapse'].dt.seconds
        test_df.loc[(test_df['elapse'] > 600) | (test_df['elapse'] < 2), 'elapse'] = np.nan
        test_df['elapse'].fillna(0)


        # 카테고리 컬럼 끝 _c 붙여주세요.
        merged_df = merged_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                # 'train_know_rates': 'train_know_rates_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                # 'train_know_rates': 'train_know_rates_c',
            }
        )
        return merged_df, test_df


class FE22(FeatureEngineer):
    def __str__(self):
        return \
            """inter 1, 2, 3, 4 추가
            testId 대분류 추가(난이도 판별됨)
            assessment 별 정답률 카테고라이즈 추가 (문항별 난이도느낌)
            user 별 푼 문제들에 대한 평균 정답률 추가
            유저, 시험별 타임 일랩스 추가
            모두 머지 적용
            테스트 데이터만으로 학습
            """
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
        # fe_num = f'[{self.__class__.__name__}]' # <- 클래스 번호 출력용.
        train_df['interaction1'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
        test_df['interaction1'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    
        train_df['interaction2'] = train_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']
        test_df['interaction2'] = test_df.groupby(['userID','testId'])[['interaction1']].shift()['interaction1']

        train_df['interaction3'] = train_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']
        test_df['interaction3'] = test_df.groupby(['userID','testId'])[['interaction2']].shift()['interaction2']

        train_df['interaction4'] = train_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']
        test_df['interaction4'] = test_df.groupby(['userID','testId'])[['interaction3']].shift()['interaction3']

        merged_df = pd.concat([train_df, test_df[test_df['answerCode'] != -1]])

        merged_df['testId_large'] = merged_df['testId'].apply(lambda s: s[2])
        test_df['testId_large'] = test_df['testId'].apply(lambda s: s[2])

        def categorize(x):
            if  x <= 0.1:
                return 0
            elif 0.1 < x <= 0.2:
                return 1
            elif 0.2 < x <= 0.3:
                return 2
            elif 0.3 < x <= 0.4:
                return 3
            elif 0.4 < x <= 0.5:
                return 4
            elif 0.5 < x <= 0.6:
                return 5
            elif 0.6 < x <= 0.7:
                return 6
            elif 0.7 < x <= 0.8:
                return 7
            elif 0.8 < x <= 0.9:
                return 8
            else:
                return 9


        merged_assessment_rates = merged_df.groupby('assessmentItemID')['answerCode'].mean()

        merged_df['assessment_rate'] = merged_df['assessmentItemID'].map(merged_assessment_rates.map(categorize))
        test_df['assessment_rate'] = test_df['assessmentItemID'].map(merged_assessment_rates.map(categorize))


        merged_user_rate = merged_df.groupby('userID')['answerCode'].mean()
        test_user_rate = test_df.groupby('userID')['answerCode'].mean()

        merged_df['user_rate'] = merged_df['userID'].map(merged_user_rate.map(categorize))
        test_df['user_rate'] = test_df['userID'].map(test_user_rate.map(categorize))
        


        merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])
        merged_df['elapse'] = merged_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
        merged_df.loc[merged_df['testId'] != merged_df['testId'].shift(-1), 'elapse'] = pd.NaT
        merged_df['elapse'] = merged_df['elapse'].dt.seconds
        merged_df.loc[(merged_df['elapse'] > 600) | (merged_df['elapse'] < 2), 'elapse'] = np.nan
        merged_df['elapse'].fillna(0)

        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
        test_df['elapse'] = test_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
        test_df.loc[test_df['testId'] != test_df['testId'].shift(-1), 'elapse'] = pd.NaT
        test_df['elapse'] = test_df['elapse'].dt.seconds
        test_df.loc[(test_df['elapse'] > 600) | (test_df['elapse'] < 2), 'elapse'] = np.nan
        test_df['elapse'].fillna(0)


        # 카테고리 컬럼 끝 _c 붙여주세요.
        merged_df = merged_df.rename(columns=
            {
                'assessmentItemID': 'assessmentItemID_c', # 기본 1
                'testId': 'testId_c', # 기본 2
                'KnowledgeTag': 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                # 'train_know_rates': 'train_know_rates_c',
            }
        )
        test_df = test_df.rename(columns=
            {
                'assessmentItemID' : 'assessmentItemID_c', # 기본 1
                'testId' : 'testId_c', # 기본 2
                'KnowledgeTag' : 'KnowledgeTag_c', # 기본 3
                'interaction1' : 'interaction1_c',
                'interaction2' : 'interaction2_c',
                'interaction3' : 'interaction3_c',
                'interaction4' : 'interaction4_c',
                'testId_large' : 'testId_large_c',
                # 'test_rate' : 'test_rate_c',
                'assessment_rate' : 'assessment_rate_c',
                'user_rate' : 'user_rate_c',
                # 'train_know_rates': 'train_know_rates_c',
            }
        )
        return test_df[test_df['answerCode'] != -1], test_df


def main():
    if not os.path.exists(BASE_DATA_PATH):
        os.mkdir(BASE_DATA_PATH)
    
    base_train_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'train_data.csv'), parse_dates=['Timestamp'])
    base_test_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'test_data.csv'), parse_dates=['Timestamp'])

    # # 클래스 생성 후 여기에 번호대로 추가해주세요.
    # FE00(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE01(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE02(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE03(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE04(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE05(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE06(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE07(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE08(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE09(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE10(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE11(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE12(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE13(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE15(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE16(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE17(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE18(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE19(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE20(BASE_DATA_PATH, base_train_df, base_test_df).run()
    # FE21(BASE_DATA_PATH, base_train_df, base_test_df).run()
    FE22(BASE_DATA_PATH, base_train_df, base_test_df).run()
if __name__=='__main__':
    main()
