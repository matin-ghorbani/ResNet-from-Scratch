import torch
from torch import nn


class Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            identity_downsample=None,
            stride=1
    ):
        super().__init__()

        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(
            self,
            layers,
            img_channels,
            num_classes
    ):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            img_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(layers[0], 64, stride=1)
        self.layer2 = self._make_layer(layers[1], 128, stride=2)
        self.layer3 = self._make_layer(layers[2], 256, stride=2)
        self.layer4 = self._make_layer(layers[3], 512, stride=2)  # 2048 channels at the end

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, num_residual, out_channels, stride):
        identity_downsample = None
        layers = nn.ModuleList()

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

        layers.append(Block(
            self.in_channels,
            out_channels,
            identity_downsample,
            stride
        ))
        self.in_channels = out_channels * 4

        for i in range(num_residual - 1):
            layers.append(Block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)

        return self.fc(x)


def resnet50(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 6, 3], img_channels, num_classes)


def resnet101(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 23, 3], img_channels, num_classes)


def resnet152(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 36, 3], img_channels, num_classes)


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = resnet152().to(device)
    x = torch.randn(2, 3, 224, 224, device=device)

    y: torch.Tensor = net(x)
    print(f'{y.size() = }')


if __name__ == '__main__':
    main()
