import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import OrdinalEncoder

from args import parse_args


class FeatureEngineer:
    def __init__(self, args):
        self.args = args
        self.cate_cols = ["answerCode", "assessmentItemID", "testId", "KnowledgeTag"]

    def load_data(self, is_train=True):

        if not os.path.exists(self.args.fe_dir):
            raise FileNotFoundError

        # load data from file
        if is_train:
            filename = 'fe_train.csv'
        else:
            filename = 'fe_test.csv'

        file_path = os.path.join(self.args.fe_dir, filename)
        df = pd.read_csv(file_path)

        return df
        

    def _feature_engineering(self, df:pd.DataFrame):

        # Change existing categorical column names
        new_names = {name:name+'C_' for name in self.cate_cols}
        df.rename(new_names, axis=1. inplace=True)

        # Update self.cate_cols
        new_features = []
        self.cate_cols = self.cate_cols.extend(new_features)

        raise NotImplementedError()


    def _label_encoding(self, df, is_train=True):
        ### 후에 Embedding 할 때, embedding layer input으로 전체 카테고리 항목 개수를 넣어줘야 해서 이 값도 저장해줘야 하는데,
        ### 그냥 integer value 하나라서 어떻게 저장하면 효율적일지 모르겠네요...
        ### (기존 Baseline에서는 모든게 한번에 진행되서 args.n_cate 이런식으로 저장해주고 나중에 불러왔어요)

        ### Ordinal Encoder랑 강의 코드에서 따온 offset encoding 둘 다 넣어놨습니다
        ### Ordinal Encoder 쓸 때는 __save_labels() 따로 정의해줘서 encoding mapping 해줄 파일 저장하는 방식을 되있습니다
        ### (그냥 self.labels로 정의해줘서 써도 가능할 것 같긴 한데 이것도 embedding layer input 값 때문에 문제네용)


        # #################### Ordinal Encoder
        # for col in self.cate_cols:

        #     oe = OrdinalEncoder()
        #     if is_train:
        #         # For UNKNOWN class
        #         a = df[col].unique().tolist() + ["unknown"]
        #         oe.fit(a)
        #         self.__save_labels(oe, col)
        #     else:
        #         label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
        #         oe.classes_ = np.load(label_path)

        #         df[col] = df[col].apply(
        #             lambda x: x if str(x) in oe.classes_ else "unknown"
        #         )

        #     df[col] = df[col].astype(str)
        #     test = oe.transform(df[col])
        #     df[col] = test

         

        # ################## Offset        
        # mappers_dict = {}

        # # nan 값이 0이므로 위해 offset은 1에서 출발한다
        # cate_offset = 1

        # for col in self.cate_cols:
            
        #     # 각 column마다 mapper를 만든다
        #     cate2idx = {}
        #     for v in df[col].unique():

        #         # nan 및 None은 넘기는 코드
        #         if (v != v) | (v == None):
        #             continue

        #         # offset 추가 - cumulative sum
        #         cate2idx[v] = len(cate2idx) + cate_offset 

        #     mappers_dict[col] = cate2idx 

        #     # mapping
        #     df[col] = df[col].map(cate2idx).fillna(0).astype(int)

        #     # offset 추가 - cumulative sum 
        #     cate_offset += len(cate2idx)

        # self.args.cate_offset = cate_offset

        # def convert_time(s):
        #     timestamp = time.mktime(
        #         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        #     )
        #     return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)

        # return df
        pass

        
    def split_data(self, data):
        pass

    def run(self):
        train_df = self.load_data(is_train=True)
        test_df = self.load_data(is_train=False)

        train_df = self._feature_engineering(train_df)
        test_df = self._feature_engineering(test_df)

        train_df = self._label_encoding(is_train=True)
        test_df = self._label_encoding(is_train=False)
        
        # 저장까지 FE01/

        # save 
        train_df.to_csv(os.path.join(self.args.fe_dir, 'train.csv'))
        test_df.to_csv(os.path.join(self.args.fe_dir, 'test.csv'))
        
    

class FE(FeatureEngineer):
    def __init__(self):
        super().__init__()

    def feature_engineering(data: pd.DataFrame):
        # 여기에 열심히 fe 한 코드들 복붙
        return None


if __name__=='__main__':
    new_df = FE()
    new_df.run()
