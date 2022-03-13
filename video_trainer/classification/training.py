import os

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

EPOCHS = 100

MODEL = torchvision.models.video.r2plus1d_18(pretrained=True)
DEVICE = torch.device('cuda')
SCALER = torch.cuda.amp.GradScaler()

torch.backends.cudnn.benchmark = True


def train(training_loader: DataLoader, validation_loader: DataLoader) -> None:
    MODEL.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(MODEL.parameters())
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch + 1}/{EPOCHS}')
        print('Training:')
        correct = 0
        total = 0
        running_loss = 0
        MODEL.train()
        for inputs, targets in tqdm(training_loader):
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            with torch.cuda.amp.autocast():
                outputs = MODEL(inputs)
                loss = criterion(outputs, targets)

            SCALER.scale(loss).backward()
            SCALER.step(optimizer)
            SCALER.update()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            running_loss += loss.item()
        train_loss = running_loss / len(training_loader)
        train_acc = 100 * correct / total
        print(f'Loss: {train_loss}. Accuracy: {train_acc}')
        MODEL.eval()
        with torch.inference_mode():
            print('Validation:')
            correct = 0
            total = 0
            running_loss = 0
            for inputs, labels in validation_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = MODEL(inputs)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += val_loss.item()
        val_loss = running_loss / len(validation_loader)
        val_acc = 100 * correct / total
        print(f'Loss: {val_loss}. Accuracy: {val_acc}')
        with open(os.path.join('..', 'models', f'epoch-{epoch}.p'), 'wb') as file:
            torch.save(MODEL, file)
