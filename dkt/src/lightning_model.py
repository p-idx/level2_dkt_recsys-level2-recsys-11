import os
import datetime

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC


class DKTLightning(pl.LightningModule):
    def __init__(self, args, model: nn.Module):
        super().__init__()
        self.args = args
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.auc = AUROC(pos_label=1) # 얘네가 좋은 점은 사이킷런 꺼 보다 똘똘함.
        self.acc = Accuracy(threshold=0.5)
    
        
    def training_step(self, batch: tuple, batch_idx) -> torch.Tensor:
        cate_x, cont_x, mask, targets = batch
        preds = self.model(cate_x, cont_x, mask)

        losses = self.loss_fn(preds, targets)[:, -1]
        loss = torch.mean(losses)

        self.auc(preds[:, -1], targets[:, -1].long())
        self.acc(preds[:, -1], targets[:, -1].long())

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_auc', self.auc) # __str__ 오버라이딩을 통해 그냥 들어가도 되는 듯.
        self.log('train_acc', self.acc)
        return loss


    def configure_optimizers(self):
        # Adam 말고 나중에 다른걸로
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer
        
    
    def validation_step(self, batch, batch_idx):
        cate_x, cont_x, mask, targets = batch
        preds = self.model(cate_x, cont_x, mask)
        val_losses = self.loss_fn(preds, targets)[:, -1] # 맨 끝 many-to-one
        val_loss = torch.mean(val_losses) # 배치들을 평균

        self.auc(preds[:, -1], targets[:, -1].long())
        self.acc(preds[:, -1], targets[:, -1].long())

        self.log('valid_loss', val_loss)
        self.log('valid_auc', self.auc) # __str__ 오버라이딩을 통해 그냥 들어가도 되는 듯.
        self.log('valid_acc', self.acc)
        return val_loss
    
    
    def predict_step(self, batch, batch_idx: int):
        cate_x, cont_x, mask, targets = batch
        preds = self.model(cate_x, cont_x, mask)
        preds = torch.sigmoid(preds[:, -1]) # 안해줘도 제출 시 거기서도 torchmetric 으로 할것같은 느낌임.
        return preds.detach().cpu()
    

    def on_predict_epoch_end(self, results):
        time_info = (datetime.datetime.today() + datetime.timedelta(hours= 9)).strftime('%m%d_%H%M')
        write_path = os.path.join(
            self.args.output_dir, 
            f"{self.model.__class__.__name__}_{time_info}.csv"
        )

        total_preds = torch.cat(results[0]).tolist()

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write("{},{}\n".format(id, p))
        
        return total_preds
