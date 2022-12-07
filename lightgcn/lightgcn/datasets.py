import os

import pandas as pd
import torch


def prepare_dataset(device, basepath, fe_num, verbose=True, logger=None):
    # data = load_data(basepath)
    # train_data, test_data = separate_data(data)
    train_data, test_data = load_data(basepath, fe_num)

    # 전체 유저와 문제 인덱싱
    id2index = indexing_data(train_data, test_data)

    # train, valid, test split
    train_data, valid_data, test_data = separate_data(train_data, test_data)

    # edge와 label 정의
    train_data_proc = process_data(train_data, id2index, device)
    valid_data_proc = process_data(valid_data, id2index, device, is_valid=True)
    test_data_proc = process_data(test_data, id2index, device)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(valid_data, 'Valid', logger=logger)
        print_data_stat(test_data, "Test", logger=logger)
    

    return train_data_proc, valid_data_proc, test_data_proc, len(id2index)


def load_data(basepath, fe_num):
    # path1 = os.path.join(basepath, "train_data.csv")
    # path2 = os.path.join(basepath, "test_data.csv")
    # data1 = pd.read_csv(path1)
    # data2 = pd.read_csv(path2)

    # data = pd.concat([data1, data2])

    path1 = os.path.join(basepath, f"FE{fe_num}", "train_data.csv")
    path2 = os.path.join(basepath, f"FE{fe_num}", "test_data.csv")
    train = pd.read_csv(path1) # merged_train
    test = pd.read_csv(path2)

    train.drop_duplicates(
        subset=["userID", "assessmentItemID_c"], keep="last", inplace=True
    )

    return train, test 


def separate_data(train, test):
    # train_data = data[data.answerCode >= 0]
    # test_data = data[data.answerCode < 0]

    valid = train.groupby('userID').tail(3)
    train = train.drop(index=valid.index)
    valid = valid.reset_index(drop=True)

    test = test[test.answerCode == -1]

    return train, valid, test 


def indexing_data(train, test):
    data = pd.concat([train,test])

    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID_c))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {str(v): i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(data, id_2_index, device, is_valid=False):
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID_c, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[str(item)]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)

    if is_valid:
        return dict(edge=edge.to(device), label=label.numpy())
    
    else:
        return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID_c))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
