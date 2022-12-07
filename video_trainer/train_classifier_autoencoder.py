from dataclasses import dataclass

import pytorch_lightning as lightning
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from video_trainer import models, settings, system_classifier_autoencoder
from video_trainer.loading.data_module_classification import DataModuleClassification


@dataclass
class ClassifierConfigSet:
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    classifier: torch.nn.Module
    encoder_model_dir: str


CONFIG = {
    '2_layer': ClassifierConfigSet(
        encoder=models.Encoder2Layer,
        decoder=models.Decoder2Layer,
        classifier=models.Classifier2Layer,
        encoder_model_dir='./mlruns/0/'
        '19a5ed219a634eb0b41c007b588cc014/checkpoints/epoch=89-step=250200.ckpt',
    ),
    '3_layer': ClassifierConfigSet(
        encoder=models.Encoder3Layer,
        decoder=models.Decoder3Layer,
        classifier=models.Classifier3Layer,
        encoder_model_dir='./mlruns/0/'
        'd476faca3c204772ae5a0e56c4894197/checkpoints/epoch=63-step=177920.ckpt',
    ),
    '2_layer_rgb': ClassifierConfigSet(
        encoder=models.Encoder2LayerRGB,
        decoder=models.Decoder2LayerRGB,
        classifier=models.Classifier2LayerRGB,
        encoder_model_dir='./mlruns/0/'
        '929e74df23d8442e994e3b479c1490be/checkpoints/epoch=54-step=152900.ckpt',
    ),
    '3_layer_rgb': ClassifierConfigSet(
        encoder=models.Encoder3LayerRGB,
        decoder=models.Decoder3LayerRGB,
        classifier=models.Classifier3LayerRGB,
        encoder_model_dir='./mlruns/0/'
        '337bc7b6d9f041039f5a3b970dfbbcc9/checkpoints/epoch=57-step=161240.ckpt',
    ),
    '3_layer_rgb_reduced': ClassifierConfigSet(
        encoder=models.Encoder3LayerReducedTimeStride,
        decoder=models.Decoder3LayerReducedTimeStride,
        classifier=models.Classifier3LayerReducedTimeStride,
        encoder_model_dir='',
    ),
    '3_layer_rgb_linear_256': ClassifierConfigSet(
        encoder=models.Encoder2LayerRGBLinear256,
        decoder=models.Decoder2LayerRGBLinear256,
        classifier=models.Classifier2LayerRGBLinear256,
        encoder_model_dir='./mlruns/0/'
        'b8b335c646344255877b59b60d89678c/checkpoints/epoch=24-step=69500.ckpt',
    ),
    '3_layer_rgb_linear_1024_256': ClassifierConfigSet(
        encoder=models.Encoder2LayerRGBLinear1024_256,
        decoder=models.Decoder2LayerRGBLinear1024_256,
        classifier=models.Classifier2LayerRGBLinear_1024_256,
        encoder_model_dir='./mlruns/0/'
        'b8b0e177342540719af9247379750a0a/checkpoints/epoch=198-step=553220.ckpt',
    ),
}


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
        model=system_classifier_autoencoder.System(CONFIG['3_layer_rgb_linear_1024_256']),
        datamodule=data_module_classification,
    )


if __name__ == '__main__':
    main()
