import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

EPOCHS = 2

MODEL = torchvision.models.video.r2plus1d_18(pretrained=True)
DEVICE = torch.device('cuda')


def train(training_dataset: DataLoader, validation_dataset: DataLoader) -> None:
    MODEL.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(MODEL.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    for _ in range(EPOCHS):
        for video_batch, label in tqdm(training_dataset):
            MODEL.train()
            video, target = video_batch.to(DEVICE), label.to(DEVICE)
            output = MODEL(video)
            loss = criterion(output, target)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
        MODEL.eval()
        with torch.inference_mode():
            for video, target in validation_dataset:
                video = video.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                output = MODEL(video)
                loss = criterion(output, target)
