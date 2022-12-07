from dataclasses import dataclass

import pytorch_lightning as lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from video_trainer import settings
from video_trainer.loading.data_module_classification import DataModuleClassification
from video_trainer.system import System


@dataclass
class ClassifierConfigSet:
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    classifier: torch.nn.Module
    encoder_model_dir: str


def main() -> None:

    data_module_classification = DataModuleClassification()

    trainer = lightning.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=settings.EPOCHS,
        precision=16,
        logger=MLFlowLogger(experiment_name='Default'),
        callbacks=[ModelCheckpoint(save_top_k=3, monitor='val_loss')],
    )

    trainer.fit(
        model=System(),
        datamodule=data_module_classification,
    )


if __name__ == '__main__':
    main()
