import os
import random

import numpy as np
import torch


def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def transform_proba(pred:list):
    output = []
    max_idx = np.argmax(pred, axis=1)
    for i,v in enumerate(max_idx):
        if v == 0:
            output.append(1 - pred[i][v])
        else:
            output.append(pred[i][v])
    return output