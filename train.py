import subprocess
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import torch.nn.functional as F
from options import options as opt

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

# de_id → 分类器类别索引的映射
# 3=rain/raindrop, 4=fog/haze, 6=snow，映射到 0~2
# 其他未知值安全归到 0，不会崩溃
DE_ID_REMAP = {3: 0, 4: 1, 6: 2}

class PromptIRModel(pl.LightningModule):
    def __init__(self, lambda_cls=0.1, experiment_name=""):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.lambda_cls = float(lambda_cls)
        self.experiment_name = str(experiment_name)
    
    def forward(self, x, tta_z=None):
        return self.net(x, tta_z=tta_z)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        
        # 主损失：图像复原 L1
        restored = self.net(degrad_patch)
        loss_l1 = self.loss_fn(restored, clean_patch)

        # 辅助损失：天气分类监督
        _, logits = self.net.weather_encoder(degrad_patch, return_logits=True)
        de_id_mapped = torch.tensor(
            [DE_ID_REMAP.get(int(i), 0) for i in de_id],
            dtype=torch.long, device=degrad_patch.device
        )
        loss_cls = F.cross_entropy(logits, de_id_mapped)

        weighted_loss_cls = self.lambda_cls * loss_cls
        loss = loss_l1 + weighted_loss_cls
        
        self.log("train_loss", loss)
        self.log("loss_l1", loss_l1)
        self.log("loss_cls", loss_cls)
        self.log("weighted_loss_cls", weighted_loss_cls)
        self.log("lambda_cls", self.lambda_cls)
        return loss
    
    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=200
        )

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    
    experiment_name = str(getattr(opt, "experiment_name", "")).strip()
    if experiment_name:
        logger = TensorBoardLogger(save_dir=opt.log_dir, name=experiment_name)
    else:
        logger = TensorBoardLogger(save_dir=opt.log_dir)

    trainset = PromptTrainDataset(opt)
