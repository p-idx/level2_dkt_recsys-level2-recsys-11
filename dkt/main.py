import os
import numpy as np
import torch
import wandb

from args import parse_args
from src import trainer
from src.dataloader_ import get_data, split_data, get_loader
from src.utils import setSeeds
from src.model_ import LSTM

import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

def main(args):
    # wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 일단 스플릿 까지 하고.
    train_data, offset, cate_num, cont_num = get_data(args, is_train=True)
    train_data, valid_data = split_data(train_data)
    test_data = get_data(args, is_train=False)

    args.offset = offset + 1
    args.cate_num = cate_num
    args.cont_num = cont_num
    train_dataloader, valid_dataloader = get_loader(args, train_data, valid_data)
    _, test_dataloader = get_loader(args, None, test_data)

    # train test
    model = LSTM(args).to(device=args.device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    def get_metric(targets, preds):
        auc = roc_auc_score(targets, preds)
        acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))

        return auc, acc

    for epoch in range(args.n_epochs):
        total_loss = []
        total_preds = []
        total_targets = []
        
        model.train()
        for cate_x, cont_x, mask, targets in train_dataloader:
            cate_x = cate_x.to(args.device)
            cont_x = cont_x.to(args.device)
            mask = mask.to(args.device)
            targets = targets.to(args.device)
            
            preds = model(cate_x, cont_x, mask)

            losses = criterion(preds, targets)
            loss = torch.mean(losses[:, -1])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            total_loss.append(loss.item())
            total_preds.append(preds[:, -1].detach())
            total_targets.append(targets[:, -1].detach())

        total_preds = torch.concat(total_preds).cpu().numpy()
        total_targets = torch.concat(total_targets).cpu().long().numpy()


        # Train AUC / ACC
        auc, acc = get_metric(total_targets, total_preds)
        loss_avg = np.mean(total_loss)
        # print(total_preds)
        if epoch % args.log_steps == 0:
            print(f"[{epoch}] TRAIN AUC : {auc} ACC : {acc} loss: {loss_avg}")



        total_preds = []
        total_targets = []
        
        model.eval()
        for cate_x, cont_x, mask, targets in valid_dataloader:
            cate_x = cate_x.to(args.device)
            cont_x = cont_x.to(args.device)
            mask = mask.to(args.device)
            targets = targets.to(args.device)
            
            preds = model(cate_x, cont_x, mask)

            total_preds.append(preds[:, -1].detach())
            total_targets.append(targets[:, -1].detach())

        total_preds = torch.concat(total_preds).cpu().numpy()
        total_targets = torch.concat(total_targets).cpu().long().numpy()

        # Valid AUC / ACC
        auc, acc = get_metric(total_targets, total_preds)
        if epoch % args.log_steps == 0:
            print(f"[{epoch}] VALID AUC : {auc} ACC : {acc}")


    total_preds = []
    model.eval()
    for cate_x, cont_x, mask, targets in test_dataloader:
        cate_x = cate_x.to(args.device)
        cont_x = cont_x.to(args.device)
        mask = mask.to(args.device)
        targets = targets.to(args.device)
        
        preds = model(cate_x, cont_x, mask)
        preds = preds[:, -1]
        preds = nn.Sigmoid()(preds)
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)
    
    # 타임, fe, 모델명 추가.
    write_path = os.path.join(args.output_dir, "submission-t.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
