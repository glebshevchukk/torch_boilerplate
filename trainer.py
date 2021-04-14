
from torch_boilerplate import *

from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback, ModelCheckpoint
import argparse
import wandb

def run_trainer():
    wandb.init()
    config = wandb.config

    model = None
    data = None
    cbs = []

    checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename=config.exp_name,
            verbose=True,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
    )
    cbs.append(checkpoint_callback)
    
    wandb_logger = WandbLogger(project=config.exp_name)
    trainer=pl.Trainer(
        max_epochs=config.num_epochs,
        gpus=config.gpus,
        progress_bar_refresh_rate=1000,
        logger=wandb_logger,
        callbacks = cbs
    )
    
    trainer.fit(model, data)

run_trainer()