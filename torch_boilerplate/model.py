import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class SupervisedModel(pl.LightningModule):
    def __init__(self):
        super(SupervisedModel, self).__init__()
        self.net = None
        self.loss = None

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb) :
        x_train,y_train = batch
    
        out = self.forward(x_train)
        loss = self.loss(out,y_train)
        
        return {'loss':loss}
    
    def validation_step(self, batch, batch_nb) :
        x_val,y_val = batch
    
        out = self.forward(x_val)
        loss = self.loss(out,y_val)
        
        return {'val_loss':loss}

    def test_step(self, batch, batch_nb) :
        x_test,y_test = batch
    
        out = self.forward(x_test)
        loss = self.loss(out,y_test)
        
        return {'test_loss':loss}


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters())
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]
    
    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return train_loader
    
    def test_dataloader(self):
        return test_loader