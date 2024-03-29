import torch
import torchvision
from torch.nn import Conv3d, Flatten, Linear, ReLU, Sigmoid, Unflatten
from torch.nn.modules.conv import ConvTranspose3d


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


class MVitV2FineTuned(torch.nn.Module):
    def __init__(self, num_classes: int = 5, has_frozen_weights: bool = True):
        super().__init__()
        self.name = 'mvit_v2_s'
        self.frozen_weights = has_frozen_weights
        self.model = torchvision.models.video.mvit_v2_s(weights='DEFAULT')
        if has_frozen_weights:
            self._set_parameter_requires_grad()
        in_features = int(self.model.head[1].in_features)
        self.model.head[1] = torch.nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _set_parameter_requires_grad(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False


class Encoder3LayerRGB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder3LayerRGB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier3LayerRGB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Flatten(),
            Linear(in_features=1176, out_features=5),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder2LayerRGB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder2LayerRGB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier2LayerRGB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Flatten(),
            Linear(in_features=9408, out_features=5),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderWide(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=32, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=32, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderWide(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            ConvTranspose3d(in_channels=3, out_channels=32, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=32, out_channels=3, kernel_size=4, padding=1, stride=2),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder2Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=2, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=2, out_channels=1, kernel_size=4, padding=1, stride=2),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder2Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            ConvTranspose3d(in_channels=1, out_channels=2, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=2, out_channels=3, kernel_size=4, padding=1, stride=2),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier2Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Flatten(),
            Linear(in_features=3136, out_features=5),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder3Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=2, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=2, out_channels=1, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=1, out_channels=1, kernel_size=4, padding=1, stride=2),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder3Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=1, out_channels=2, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=2, out_channels=3, kernel_size=4, padding=1, stride=2),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier3Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Flatten(),
            Linear(in_features=392, out_features=5),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder3LayerReducedTimeStride(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=2, kernel_size=4, padding=1, stride=(2, 2, 2)),
            ReLU(),
            Conv3d(in_channels=2, out_channels=1, kernel_size=4, padding=1, stride=(1, 2, 2)),
            ReLU(),
            Conv3d(in_channels=1, out_channels=1, kernel_size=4, padding=1, stride=(1, 2, 2)),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder3LayerReducedTimeStride(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            ConvTranspose3d(
                in_channels=1, out_channels=1, kernel_size=4, padding=1, stride=(1, 2, 2)
            ),
            ReLU(),
            ConvTranspose3d(
                in_channels=1, out_channels=2, kernel_size=4, padding=1, stride=(1, 2, 2)
            ),
            ReLU(),
            ConvTranspose3d(
                in_channels=2, out_channels=3, kernel_size=4, padding=1, stride=(2, 2, 2)
            ),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder2LayerReducedTimeStride(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=2, kernel_size=4, padding=1, stride=(2, 2, 2)),
            ReLU(),
            Conv3d(in_channels=2, out_channels=1, kernel_size=4, padding=1, stride=(1, 2, 2)),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder2LayerReducedTimeStride(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            ReLU(),
            ConvTranspose3d(
                in_channels=1, out_channels=2, kernel_size=4, padding=1, stride=(1, 2, 2)
            ),
            ReLU(),
            ConvTranspose3d(
                in_channels=2, out_channels=3, kernel_size=4, padding=1, stride=(2, 2, 2)
            ),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier3LayerReducedTimeStride(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Flatten(),
            Linear(in_features=72, out_features=5),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder2LayerRGBLinear256(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Flatten(),
            Linear(in_features=9408, out_features=256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder2LayerRGBLinear256(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Linear(in_features=256, out_features=9408),
            Unflatten(1, [3, 4, 28, 28]),
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier2LayerRGBLinear256(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Linear(in_features=256, out_features=5),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder2LayerRGBLinear1024_256(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Conv3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            Flatten(),
            Linear(in_features=9408, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder2LayerRGBLinear1024_256(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Linear(in_features=256, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=9408),
            Unflatten(1, [3, 4, 28, 28]),
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            ReLU(),
            ConvTranspose3d(in_channels=3, out_channels=3, kernel_size=4, padding=1, stride=2),
            Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Classifier2LayerRGBLinear_1024_256(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            Linear(in_features=256, out_features=5),
            ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
