import torch
import torch.nn as nn
import torchvision


def get_model_alpha(model_name):
    if model_name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = torch.nn.Linear(512, 260)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34()
        model.fc = torch.nn.Linear(512, 260)
    elif model_name == "myresnet18":
        # 修改自定义 ResNet18
        model = ResNet18_alpha(3, 260)
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")
    return model


class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SimpleResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.dowmsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.dowmsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.dowmsample is not None:
            identity = self.dowmsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18_alpha(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet18_alpha, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [SimpleResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(SimpleResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
