import pytorch_lightning as lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from video_trainer import settings
from video_trainer.loading.dataloader import create_dataloaders
from video_trainer.loading.dataset import create_datasets
from video_trainer.system import System


def main() -> None:

    dataset_train, dataset_validation = create_datasets()
    dataloader_train, dataloader_validation = create_dataloaders(dataset_train, dataset_validation)

    trainer = lightning.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=settings.EPOCHS,
        precision=16,
        logger=MLFlowLogger(experiment_name='Default'),
        callbacks=[ModelCheckpoint(save_top_k=3, monitor='val_acc')],
    )

    trainer.fit(
        model=System(),
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_validation,
    )


if __name__ == '__main__':
    main()
