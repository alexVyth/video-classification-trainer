from torch.utils.data import DataLoader

EPOCHS = 10


def train(training_dataset: DataLoader, validation_dataset: DataLoader) -> None:
    for _ in range(EPOCHS):
        for video_batch, label in training_dataset:
            pass
