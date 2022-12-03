import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")
from typing import Tuple

from sklearn.preprocessing import OrdinalEncoder

BASE_DATA_PATH = '/opt/ml/level2_dkt_recsys-level2-recsys-11/data/'

class EDA:
    def __init__(self, is_merge: bool=True, is_new: bool=False):
        self.is_merge = is_merge
        self.is_new = is_new

        if is_new:
            self.train_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'new_train_data.csv'))
        else:
            self.train_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'train_data.csv'))
        self.test_df = pd.read_csv(os.path.join(BASE_DATA_PATH, 'test_data.csv'))
        
        if is_merge:
            self.train_df = \
                pd.concat([self.train_df, self.test_df[self.test_df['answerCode'] != -1]])

        # eda 처리용 데이터셋들
        self.base_train_df = self.train_df.copy()
        self.base_test_df = self.test_df.copy()

        self.train_df = self.train_df[['userID', 'answerCode']]
        self.test_df = self.test_df[['userID', 'answerCode']]


    def run(self, funcs: list):
        for func in funcs:
            print(f'{func.__name__} preprocessing...')
            train_series, test_series = \
                func(self.base_train_df.copy(), self.base_test_df.copy())
            self.train_df = pd.concat([self.train_df, train_series], axis=1)
            self.test_df = pd.concat([self.test_df, test_series], axis=1)

        if self.is_new:
            self.train_df.to_csv(os.path.join(BASE_DATA_PATH, f'EDA_new_train_data.csv'), index=False)
        else:
            self.train_df.to_csv(os.path.join(BASE_DATA_PATH, f'EDA_train_data.csv'), index=False)
        self.test_df.to_csv(os.path.join(BASE_DATA_PATH, f'EDA_test_data.csv'), index=False)

        # 라벨 인코딩
        self.train_df.loc[len(self.train_df)] = np.nan
        cate_cols = [col for col in self.train_df.columns if col[-2:] == '_c']

        print('label encoding...')
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe = oe.fit(self.train_df[cate_cols])

        self.train_df[cate_cols] = oe.transform(self.train_df[cate_cols]) + 1 # np.nan 분리용
        self.test_df[cate_cols] = oe.transform(self.test_df[cate_cols]) + 1   

        self.train_df.fillna(0, inplace=True)
        self.test_df.fillna(0, inplace=True)

        self.train_df = self.train_df[:-1]

        self.train_df[cate_cols + ['userID', 'answerCode']] = \
            self.train_df[cate_cols + ['userID', 'answerCode']].astype(np.int64)
        self.test_df[cate_cols + ['userID', 'answerCode']] = \
            self.test_df[cate_cols + ['userID', 'answerCode']].astype(np.int64)

        if self.is_new:
            self.train_df.to_csv(os.path.join(BASE_DATA_PATH, f'le_EDA_new_train_data.csv'), index=False)
        else:
            self.train_df.to_csv(os.path.join(BASE_DATA_PATH, f'le_EDA_train_data.csv'), index=False)
        self.test_df.to_csv(os.path.join(BASE_DATA_PATH, f'le_EDA_test_data.csv'), index=False)
        
        if self.is_new:
            f = open(os.path.join(BASE_DATA_PATH, f'new_feature_config.txt'), 'w')
            for i, col_name in enumerate(self.train_df.columns):
                if col_name not in ['userID', 'answerCode']:
                    f.write(f'{i-2:02},{col_name}\n')
        else:
            f = open(os.path.join(BASE_DATA_PATH, f'feature_config.txt'), 'w')
            for i, col_name in enumerate(self.train_df.columns):
                if col_name not in ['userID', 'answerCode']:
                    f.write(f'{i-2:02},{col_name}\n')
        f.close()

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


def fe_00(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # diff1_c 추가
    train_df['diff1_c'] = train_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']
    test_df['diff1_c'] = test_df.groupby(['userID','testId'])[['answerCode']].shift()['answerCode']

    return train_df['diff1_c'], test_df['diff1_c']


def fe_01(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # diff2_c 추가
    train_df['diff2_c'] = train_df.groupby(['userID','testId'])[['answerCode']].shift(2)['answerCode']
    test_df['diff2_c'] = test_df.groupby(['userID','testId'])[['answerCode']].shift(2)['answerCode']

    return train_df['diff2_c'], test_df['diff2_c']


def fe_02(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # diff3_c 추가
    train_df['diff3_c'] = train_df.groupby(['userID','testId'])[['answerCode']].shift(3)['answerCode']
    test_df['diff3_c'] = test_df.groupby(['userID','testId'])[['answerCode']].shift(3)['answerCode']

    return train_df['diff3_c'], test_df['diff3_c']


def fe_03(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # diff4_c 추가
    train_df['diff4_c'] = train_df.groupby(['userID','testId'])[['answerCode']].shift(4)['answerCode']
    test_df['diff4_c'] = test_df.groupby(['userID','testId'])[['answerCode']].shift(4)['answerCode']

    return train_df['diff4_c'], test_df['diff4_c']


def fe_04(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # diff5_c 추가
    train_df['diff5_c'] = train_df.groupby(['userID','testId'])[['answerCode']].shift(5)['answerCode']
    test_df['diff5_c'] = test_df.groupby(['userID','testId'])[['answerCode']].shift(5)['answerCode']

    return train_df['diff5_c'], test_df['diff5_c']


def fe_05(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # assessmentItemID_c 추가
    train_df.rename(columns={'assessmentItemID':'assessmentItemID_c'}, inplace=True)
    test_df.rename(columns={'assessmentItemID':'assessmentItemID_c'}, inplace=True)
    return train_df['assessmentItemID_c'], test_df['assessmentItemID_c']


def fe_06(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # testId_c 추가
    train_df.rename(columns={'testId':'testId_c'}, inplace=True)
    test_df.rename(columns={'testId':'testId_c'}, inplace=True)

    return train_df['testId_c'], test_df['testId_c'] 


def fe_07(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # KnowledgeTag_c 추가
    train_df.rename(columns={'KnowledgeTag':'KnowledgeTag_c'}, inplace=True)
    test_df.rename(columns={'KnowledgeTag':'KnowledgeTag_c'}, inplace=True)

    return train_df['KnowledgeTag_c'], test_df['KnowledgeTag_c'] 


def fe_08(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # testId_large_c 추가, 문자열 앞 세번째 태그 -> 시험별 난이도 처럼 나옴
    train_df['testId_large_c'] = train_df['testId'].apply(lambda s: s[2])
    test_df['testId_large_c'] = test_df['testId'].apply(lambda s: s[2])

    return train_df['testId_large_c'], test_df['testId_large_c']


def fe_09(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # assessmentItemID_avg_rate_c 추가, 문항 별 평균 정답률 카테고라이즈 
    train_assessment_rates = train_df.groupby('assessmentItemID')['answerCode'].mean()

    train_df['assessmentItemID_avg_rate_c'] = \
        train_df['assessmentItemID'].map(train_assessment_rates.map(categorize))
    test_df['assessmentItemID_avg_rate_c'] = \
        test_df['assessmentItemID'].map(train_assessment_rates.map(categorize))

    return train_df['assessmentItemID_avg_rate_c'], test_df['assessmentItemID_avg_rate_c']


def fe_10(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # user_avg_rate_c 추가, 유저별 정답률을 카테고라이즈
    train_user_rate = train_df.groupby('userID')['answerCode'].mean()
    test_user_rate = test_df.groupby('userID')['answerCode'].mean()


    train_df['user_avg_rate_c'] = train_df['userID'].map(train_user_rate.map(categorize))
    test_df['user_avg_rate_c'] = test_df['userID'].map(test_user_rate.map(categorize))
        
    return train_df['user_avg_rate_c'], test_df['user_avg_rate_c']


def fe_11(train_df: pd.DataFrame, test_df:pd.DataFrame) -> pd.Series:
    # elapse 추가, 유저별 푼 시간, 2 초 ~ 600 초 사이로 자름, 나머지 0.0
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    train_df['user_elapse'] = train_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
    train_df.loc[train_df['testId'] != train_df['testId'].shift(-1), 'user_elapse'] = pd.NaT
    train_df['user_elapse'] = train_df['user_elapse'].dt.seconds
    train_df.loc[(train_df['user_elapse'] > 600) | (train_df['user_elapse'] < 2), 'user_elapse'] = np.nan
    train_df['user_elapse'].fillna(0)

    # train_df['user_elapse'] = train_df['assessmentItemID'].map(train_df.groupby('assessmentItemID')['user_elapse'].mean())

    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    test_df['user_elapse'] = test_df.groupby(['userID'])['Timestamp'].diff(1).shift(-1)
    test_df.loc[test_df['testId'] != test_df['testId'].shift(-1), 'user_elapse'] = pd.NaT
    test_df['user_elapse'] = test_df['user_elapse'].dt.seconds
    test_df.loc[(test_df['user_elapse'] > 600) | (test_df['user_elapse'] < 2), 'user_elapse'] = np.nan
    test_df['user_elapse'].fillna(0)

    # test_df['user_elapse'] = train_df['assessmentItemID'].map(train_df.groupby('assessmentItemID')['user_elapse'].mean())
    return train_df['user_elapse'], test_df['user_elapse']


def fe_12(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # know_avg_rates_c 추가, 태그별 정답률
    train_know_rates = train_df.groupby('KnowledgeTag')['answerCode'].mean()

    train_df['know_avg_rates_c'] = train_df['KnowledgeTag'].map(train_know_rates.map(categorize))
    test_df['know_avg_rates_c'] = test_df['KnowledgeTag'].map(train_know_rates.map(categorize))

    return train_df['know_avg_rates_c'], test_df['know_avg_rates_c']


def fe_13(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # testId_cumcount 추가, 테스트 지금껏 몇개 풀었는가

    train_df['user_test_cumcount'] = train_df.groupby(['userID', 'testId']).cumcount()
    test_df['user_test_cumcount'] = test_df.groupby(['userID', 'testId']).cumcount()

    return train_df['user_test_cumcount'], test_df['user_test_cumcount']


def fe_14(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # user 별 시험 문제 푼 시간
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    v = (train_df.groupby(['userID', 'testId'])['Timestamp'].tail(1).view('int64').values\
     - train_df.groupby(['userID', 'testId'])['Timestamp'].head(1).view('int64').values) / 1_000_000_000
    c = train_df.groupby(['userID', 'testId'])['Timestamp'].head(1)
    c.loc[:] = v  
    train_df['user_test_time'] = np.nan
    train_df.loc[c.index, 'user_test_time'] = c
    train_df['user_test_time'].fillna(method='ffill', inplace=True)
    # train_df.loc[train_df['user_test_time'] > 3600, 'user_test_time'] = 3600
    # train_df.loc[train_df['user_test_time'] < 60, 'user_test_time'] = 60
    # train_df['user_test_time'] = train_df['testId'].map(train_df.groupby(['testId'])['user_test_time'].mean())
    train_df.loc[train_df['user_test_time'] > 3600, 'user_test_time'] = 3600
    # print(train_df['user_test_time'].describe())
    train_mean = train_df['user_test_time'].mean()
    train_std = train_df['user_test_time'].std()
    train_df['user_test_time'] = (train_df['user_test_time'] - train_mean) / train_std

    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    v = (test_df.groupby(['userID', 'testId'])['Timestamp'].tail(1).view('int64').values\
     - test_df.groupby(['userID', 'testId'])['Timestamp'].head(1).view('int64').values) / 1_000_000_000
    c = test_df.groupby(['userID', 'testId'])['Timestamp'].head(1)
    c.loc[:] = v  
    test_df['user_test_time'] = np.nan
    test_df.loc[c.index, 'user_test_time'] = c
    test_df['user_test_time'].fillna(method='ffill', inplace=True)
    # test_df.loc[train_df['user_test_time'] > 3600, 'user_test_time'] = 3600
    # test_df.loc[train_df['user_test_time'] < 60, 'user_test_time'] = 60
    # test_df['user_test_time'] = test_df['testId'].map(train_df.groupby(['testId'])['user_test_time'].mean())
    test_df.loc[train_df['user_test_time'] > 3600, 'user_test_time'] = 3600

    test_df['user_test_time'] = (test_df['user_test_time'] - train_mean) / train_std
    
    return train_df['user_test_time'], test_df['user_test_time']


def fe_15(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # 유저별 테스트 정답률
    c = train_df.groupby(['userID', 'testId'])['Timestamp'].head(1)
    v = train_df.groupby(['userID', 'testId'])['answerCode'].mean()
    c.loc[:] = v.values
    train_df['user_test_avg'] = np.nan
    train_df.loc[c.index, 'user_test_avg'] = c
    train_df['user_test_avg'].fillna(method='ffill', inplace=True)
    train_df['user_test_avg'] = train_df['testId'].map(train_df.groupby(['testId'])['user_test_avg'].mean())

    c = test_df.groupby(['userID', 'testId'])['Timestamp'].head(1)
    v = test_df.groupby(['userID', 'testId'])['answerCode'].mean()
    c.loc[:] = v.values
    test_df['user_test_avg'] = np.nan
    test_df.loc[c.index, 'user_test_avg'] = c
    test_df['user_test_avg'].fillna(method='ffill', inplace=True)
    test_df['user_test_avg'] = test_df['testId'].map(test_df.groupby(['testId'])['user_test_avg'].mean())

    return train_df['user_test_avg'], test_df['user_test_avg']


def fe_16(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    train_df['days'] = train_df['Timestamp'].dt.day_name()
    days_mean = train_df.groupby('days')['answerCode'].mean()
    train_df['days_mean'] = train_df['days'].map(days_mean)
    train_df.drop('days', axis=1, inplace=True)
    
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    test_df['days'] = test_df['Timestamp'].dt.day_name()
    test_df['days_mean'] = test_df['days'].map(days_mean)
    test_df.drop('days', axis=1, inplace=True)

    return train_df['days_mean'], test_df['days_mean']


def fe_17(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
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

    tag_groupby = train_df.groupby('KnowledgeTag').agg({
        'userID': 'count',
        'answerCode': percentile
    })

    tag_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지
    tag_groupby['tag_grade'] = tag_groupby['meanAnswerRate'].apply(lambda x: grade_map(x)) # 태그 평균 정답률을 이용한 난이도 정의
    train_df['tag_grade'] = train_df['KnowledgeTag'].map(tag_groupby['tag_grade']) # train_df mapping
    test_df['tag_grade'] = test_df['KnowledgeTag'].map(tag_groupby['tag_grade']) # test_df mapping

    return train_df['tag_grade'], test_df['tag_grade']


def fe_18(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
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

    tag_groupby = train_df.groupby('KnowledgeTag').agg({
        'userID': 'count',
        'answerCode': percentile
    })
    tag_groupby.columns = ['numUsers', 'meanAnswerRate'] # groupby 집계, numUsers : 문제를 푼 유저는 몇명인지

    tag_solved_mean = tag_groupby['numUsers'].mean() # numUsers의 평균 (평균적으로 각 태그들은 몇 명에게 노출됐는가)
    tag_groupby['tag_solved'] = tag_groupby['numUsers'].apply(lambda x:int(x>tag_solved_mean)) # 태그가 많이 노출된 편인지, 아닌지 여부
    train_df['tag_solved'] = train_df['KnowledgeTag'].map(tag_groupby['tag_solved']) # merged mapping
    test_df['tag_solved'] = test_df['KnowledgeTag'].map(tag_groupby['tag_solved']) # test_df mapping

    return train_df['tag_solved'], test_df['tag_solved']
    

def fe_19(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    # diff_c 추가
    train_df['user_diff_c'] = train_df.groupby('userID')[['answerCode']].shift()['answerCode']
    test_df['user_diff_c'] = test_df.groupby('userID')[['answerCode']].shift()['answerCode']

    return train_df['user_diff_c'], test_df['user_diff_c']


def fe_20(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series or pd.DataFrame:
    def preprocess(tags):
        tag_to_id = {}
        id_to_tag = {}
        for word in tags:
            if word not in tag_to_id:
                new_id = len(tag_to_id)
                tag_to_id[word] = new_id
                id_to_tag[new_id] = word

        corpus = np.array([tag_to_id[t] for t in tags])

        return corpus, tag_to_id, id_to_tag


    def convert_one_hot(corpus, vocab_size):
        '''원핫 표현으로 변환
        :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
        :param vocab_size: 어휘 수
        :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
        '''
        N = corpus.shape[0]

        if corpus.ndim == 1:
            one_hot = np.zeros((N, vocab_size), dtype=np.int32)
            for idx, word_id in enumerate(corpus):
                one_hot[idx, word_id] = 1

        elif corpus.ndim == 2:
            C = corpus.shape[1]
            one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
            for idx_0, word_ids in enumerate(corpus):
                for idx_1, word_id in enumerate(word_ids):
                    one_hot[idx_0, idx_1, word_id] = 1

        return one_hot


    def create_co_matrix(corpus, vocab_size, window_size=1):
        '''동시발생 행렬 생성
        :param corpus: 말뭉치(단어 ID 목록)
        :param vocab_size: 태그 수
        :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
        :return: 동시발생 행렬
        '''
        corpus_size = len(corpus)
        co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

        for idx, tag_id in tqdm(enumerate(corpus), total=len(corpus)):
            for i in range(1, window_size + 1):
                left_idx = idx - i
                right_idx = idx + i

                if left_idx >= 0:
                    left_tag_id = corpus[left_idx]
                    co_matrix[tag_id, left_tag_id] += 1

                if right_idx < corpus_size:
                    right_tag_id = corpus[right_idx]
                    co_matrix[tag_id, right_tag_id] += 1

        return co_matrix


    def ppmi(C, verbose=False, eps = 1e-8):
        '''PPMI(점별 상호정보량) 생성
        :param C: 동시발생 행렬
        :param verbose: 진행 상황을 출력할지 여부
        :return:
        '''
        M = np.zeros_like(C, dtype=np.float32)
        N = np.sum(C)
        S = np.sum(C, axis=0)
        total = C.shape[0] * C.shape[1]
        cnt = 0

        for i in tqdm(range(C.shape[0])):
            for j in range(C.shape[1]):
                pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
                M[i, j] = max(0, pmi)

                if verbose:
                    cnt += 1
                    if cnt % (total//100 + 1) == 0:
                        print('%.1f%% 완료' % (100*cnt/total))
        return M

    knowledges = np.concatenate(train_df.groupby('userID')['KnowledgeTag'].unique().values)
    corpus, tag_to_id, id_to_tag = preprocess(knowledges)
    tag_size = len(tag_to_id)
    C = create_co_matrix(corpus, tag_size, window_size=5)
    W = ppmi(C)
    U, S, V = np.linalg.svd(W)
    U2 = U[:, :16]

    for_train = pd.DataFrame(np.vstack(train_df['KnowledgeTag'].apply(lambda x: U2[tag_to_id[x]]).values), 
        columns=[f'know_tag_emb{i}' for i in range(U2.shape[1])])
    

    for_test = pd.DataFrame(np.vstack(test_df['KnowledgeTag'].apply(lambda x: U2[tag_to_id[x]]).values), 
        columns=[f'know_tag_emb{i}' for i in range(U2.shape[1])])

    return for_train, for_test


def fe_21(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series or pd.DataFrame:
    def preprocess(tags):
        tag_to_id = {}
        id_to_tag = {}
        for word in tags:
            if word not in tag_to_id:
                new_id = len(tag_to_id)
                tag_to_id[word] = new_id
                id_to_tag[new_id] = word

        corpus = np.array([tag_to_id[t] for t in tags])

        return corpus, tag_to_id, id_to_tag


    def convert_one_hot(corpus, vocab_size):
        '''원핫 표현으로 변환
        :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
        :param vocab_size: 어휘 수
        :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
        '''
        N = corpus.shape[0]

        if corpus.ndim == 1:
            one_hot = np.zeros((N, vocab_size), dtype=np.int32)
            for idx, word_id in enumerate(corpus):
                one_hot[idx, word_id] = 1

        elif corpus.ndim == 2:
            C = corpus.shape[1]
            one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
            for idx_0, word_ids in enumerate(corpus):
                for idx_1, word_id in enumerate(word_ids):
                    one_hot[idx_0, idx_1, word_id] = 1

        return one_hot


    def create_co_matrix(corpus, vocab_size, window_size=1):
        '''동시발생 행렬 생성
        :param corpus: 말뭉치(단어 ID 목록)
        :param vocab_size: 태그 수
        :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
        :return: 동시발생 행렬
        '''
        corpus_size = len(corpus)
        co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

        for idx, tag_id in tqdm(enumerate(corpus), total=len(corpus)):
            for i in range(1, window_size + 1):
                left_idx = idx - i
                right_idx = idx + i

                if left_idx >= 0:
                    left_tag_id = corpus[left_idx]
                    co_matrix[tag_id, left_tag_id] += 1

                if right_idx < corpus_size:
                    right_tag_id = corpus[right_idx]
                    co_matrix[tag_id, right_tag_id] += 1

        return co_matrix


    def ppmi(C, verbose=False, eps = 1e-8):
        '''PPMI(점별 상호정보량) 생성
        :param C: 동시발생 행렬
        :param verbose: 진행 상황을 출력할지 여부
        :return:
        '''
        M = np.zeros_like(C, dtype=np.float32)
        N = np.sum(C)
        S = np.sum(C, axis=0)
        total = C.shape[0] * C.shape[1]
        cnt = 0

        for i in tqdm(range(C.shape[0])):
            for j in range(C.shape[1]):
                pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
                M[i, j] = max(0, pmi)

                if verbose:
                    cnt += 1
                    if cnt % (total//100 + 1) == 0:
                        print('%.1f%% 완료' % (100*cnt/total))
        return M

    tests = np.concatenate(train_df.groupby('userID')['testId'].unique().values)
    corpus, tag_to_id, id_to_tag = preprocess(tests)
    tag_size = len(tag_to_id)
    C = create_co_matrix(corpus, tag_size, window_size=5)
    W = ppmi(C)
    U, S, V = np.linalg.svd(W)
    U2 = U[:, :8]

    for_train = pd.DataFrame(np.vstack(train_df['testId'].apply(lambda x: U2[tag_to_id[x]]).values), 
        columns=[f'test_emb{i}' for i in range(U2.shape[1])])
    

    for_test = pd.DataFrame(np.vstack(test_df['testId'].apply(lambda x: U2[tag_to_id[x]]).values), 
        columns=[f'test_emb{i}' for i in range(U2.shape[1])])

    return for_train, for_test


def main():
    eda = EDA(is_merge=False, is_new=True)
    funcs = [
        fe_00,
        fe_01,
        fe_02,
        fe_03,
        fe_04,
        fe_05,
        fe_06,
        fe_07,
        fe_08,
        fe_09,
        fe_10,
        fe_11,
        fe_12,
        fe_13,
        fe_14,
        fe_15,
        fe_16,
        fe_17,
        fe_18,
        fe_19,
        fe_20,
        fe_21
    ]
    eda.run(funcs)


if __name__ == '__main__':
    main()