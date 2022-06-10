import torch
import torchvision


class R2Plus1DFineTuned(torch.nn.Module):
    def __init__(self, num_classes: int = 5, has_frozen_weights: bool = True):
        super().__init__()
        self.name = 'r2plus1d_18'
        self.frozen_weights = has_frozen_weights
        self.model = torchvision.models.video.r2plus1d_18(pretrained=True)
        if has_frozen_weights:
            self._set_parameter_requires_grad()
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _set_parameter_requires_grad(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
