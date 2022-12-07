import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import roc_auc_score, accuracy_score


class DKTLightning(pl.LightningModule):
    def __init__(self, config, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        # mse 로 갈아낄수도 있음.
        if config.loss == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif config.loss == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')

        self.epoch_train_preds = []
        self.epoch_train_targets = []
        self.epoch_valid_preds = []
        self.epoch_valid_targets = []
    
        
    def training_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        cate_x, cont_x, mask, targets = batch
        preds = self.model(cate_x, cont_x, mask, targets)

    
        loss = torch.mean(self.loss_fn(preds[:, -1], targets[:, -1]))

        # loss = torch.mean(self.loss_fn(masked_preds, masked_targets))

        self.epoch_train_preds.extend(preds.clone().detach().cpu().numpy()[:, -1])
        self.epoch_train_targets.extend(targets.clone().detach().cpu().numpy()[:, -1])


        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    
    def on_train_epoch_end(self) -> None:
        epoch_train_preds = np.array(self.epoch_train_preds)
        epoch_train_targets = np.array(self.epoch_train_targets, dtype=np.int64)
        epoch_train_auc = roc_auc_score(epoch_train_targets, epoch_train_preds)
        epoch_train_acc = accuracy_score(epoch_train_targets, np.where(epoch_train_preds >= 0.5 , 1, 0))

        self.log('train_auc', epoch_train_auc, on_step=False, on_epoch=True) # __str__ 오버라이딩을 통해 그냥 들어가도 되는 듯.
        self.log('train_acc', epoch_train_acc, on_step=False, on_epoch=True)

        self.epoch_train_preds = []
        self.epoch_train_targets = []


    def configure_optimizers(self):
        # Adam 말고 나중에 다른걸로
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5, mode="max", verbose=True,
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": scheduler,
            'monitor': 'valid_auc' # 다른거 아무거나 쓰면 안됨. 실행시 뭐만 쓸수있다 알아서 익셉션줌. 굳
        }
        
    
    def validation_step(self, batch, batch_idx):
        cate_x, cont_x, mask, targets = batch
        preds = self.model(cate_x, cont_x, mask, targets)

        val_loss = torch.mean(self.loss_fn(preds[:, -1], targets[:, -1]))

        self.epoch_valid_preds.extend(preds.clone().detach().cpu().numpy()[:, -1])
        self.epoch_valid_targets.extend(targets.clone().detach().cpu().numpy()[:, -1])

        self.log('valid_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss


    def on_validation_epoch_end(self) -> None:
        epoch_valid_preds = np.array(self.epoch_valid_preds)
        epoch_valid_targets = np.array(self.epoch_valid_targets, dtype=np.int64)

        epoch_valid_auc = roc_auc_score(epoch_valid_targets, epoch_valid_preds)
        epoch_valid_acc = accuracy_score(epoch_valid_targets, np.where(epoch_valid_preds >= 0.5 , 1, 0))

        self.log('valid_auc', epoch_valid_auc, on_step=False, on_epoch=True, prog_bar=True) # __str__ 오버라이딩을 통해 그냥 들어가도 되는 듯.
        self.log('valid_acc', epoch_valid_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.epoch_valid_preds = []
        self.epoch_valid_targets = []
    
    
    def predict_step(self, batch, batch_idx: int):
        cate_x, cont_x, mask, targets = batch
        preds = self.model(cate_x, cont_x, mask, targets)
        preds = torch.sigmoid(preds[:, -1]) # 안해줘도 제출 시 거기서도 torchmetric 으로 할것같은 느낌임.
        return preds.detach().cpu()
    

    def on_predict_epoch_end(self, results):

        write_path = os.path.join(
            self.config.output_dir, 
            f"{self.config.model}_{self.config.time_info}_K{self.config.k_i}_FE{self.config.cate_cols + self.config.cont_cols}.csv"
        )

        total_preds = torch.cat(results[0]).numpy()

        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write("{},{}\n".format(id, p))
        
        return total_preds
