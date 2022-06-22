import pytorch_lightning as lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from video_trainer import settings
from video_trainer.loading.data_module_autoencoder import DataModuleAutoEncoding
from video_trainer.system_autoencoder import Autoencoder


def main() -> None:

    data_module_classification = DataModuleAutoEncoding()

    trainer = lightning.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=settings.EPOCHS,
        precision=16,
        logger=MLFlowLogger(experiment_name='Default'),
        callbacks=[ModelCheckpoint(save_top_k=3, monitor='val_loss')],
    )

    trainer.fit(
        model=Autoencoder(),
        datamodule=data_module_classification,
    )


if __name__ == '__main__':
    main()
