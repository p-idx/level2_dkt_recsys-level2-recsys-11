import pandas as pd
import torch
from config import CFG
from src.datasets import prepare_dataset
from src.utils import class2dict

use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_availble
device = torch.device("cuda" if use_cuda else "cpu")
# for specific gpu index,
# device = torch.device(f'cuda:{CFG.gpu}' if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def main():
    pass

if __name__ == "__main__":
    params = class2dict(CFG)
    
    print('1. Loading Dataset...')
    constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, valid_loader, mask, test_ground_truth_list, interacted_items = prepare_dataset(device, CFG.basepath, CFG.fe_num, CFG.batch_size, CFG.num_workers, CFG.ii_neighbor_num)


    
    main()