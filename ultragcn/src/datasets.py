import os
import pandas as pd
import numpy as np
import torch 
import torch.utils.data as data
import scipy.sparse as sp



def prepare_dataset(device, params, ii_neighbor_num):
    train_data, test_data = load_data(params['basepath'], params['fe_num'])

    # 전체 유저와 문제 인덱싱
    id2index, n_user, m_item = indexing_data(train_data, test_data)

    # train, valid, test split
    train_data, valid_data, test_data = separate_data(train_data, test_data)

    # edge, label, matrix 정의
    train_edge, train_label, valid_edge, valid_label, test_edge, train_mat, constraint_mat = process_data(train_data, valid_data, test_data, id2index, n_user, m_item)

    # Dataloader
    batch_size = params['batch_sisze']
    num_workers = params['num_workers']
    train_loader = data.DataLoader(train_edge, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(valid_edge, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = data.DataLoader(test_edge, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    # mask matrix for testing to accelerate testing speed
    # when doing topk 
    # mask = torch.zeros(n_user, m_item)
    # interacted_items = [[] for _ in range(n_user)]
    # for (u,i) in train_data:
    #     mask[u][i] = -np.inf
    #     interacted_items[u].append(i)


    # test user-item interaction, which is ground truth
    # test_ground_truth_list = [[] for _ in range(n_user)]
    # for (u, i) in test_data:
    #     test_ground_truth_list[u].append(i)

    interacted_items = train_label
    test_ground_truth_list = valid_label

    # Compute \Omega to extend UltraGCN to the item-item occurence graph
    ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, ii_neighbor_num)


    return constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, valid_loader, test_ground_truth_list, interacted_items


def load_data(basepath, fe_num):
    path1 = os.path.join(basepath, f"FE{fe_num}", "train_data.csv")
    path2 = os.path.join(basepath, f"FE{fe_num}", "test_data.csv")
    train = pd.read_csv(path1) # merged_train
    test = pd.read_csv(path2)

    train.drop_duplicates(
        subset=["userID", "assessmentItemID_c"], keep="last", inplace=True
    )

    return train, test


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

    return id_2_index, n_user, n_item


def separate_data(train, test):
    valid = train.groupby('userID').tail(1)
    train = train.drop(index=valid.index)
    valid = valid.reset_index(drop=True)

    test = test[test.answerCode == -1]

    return train, valid, test


def process_data(train_data, valid_data, test_data, id_2_index, n_user, m_item):
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32) 
    train_edge = []; valid_edge = []; test_edge = []
    train_label = []; valid_label = []
    
    # generate edges 
    for user, item, acode in zip(train_data.userID, train_data.assessmentItemID_c, train_data.answerCode):
        uid, iid = id_2_index[user], id_2_index[str(item)]
        train_edge.append([uid, iid])
        train_mat[uid, iid] = float(acode)
        train_label.append(acode)

    for user, item, acode in zip(valid_data.userID, valid_data.assessmentItemID_c, valid_data.answerCode):
        uid, iid = id_2_index[user], id_2_index[str(item)]
        valid_edge.append([uid, iid])
        valid_label.append(acode)

    for user, item, acode in zip(test_data.userID, test_data.assessmentItemID_c, test_data.answerCode):
        uid, iid = id_2_index[user], id_2_index[str(item)]
        test_edge.append([uid, iid])
    

    # construct degree matrix for graphmf
    items_D = np.sum(train_mat, axis=0).reshape(-1)
    users_D = np.sum(train_mat, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    
    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}
    
    return train_edge, train_label, valid_edge, valid_label, test_edge, train_mat, constraint_mat


def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float()