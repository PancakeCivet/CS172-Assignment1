"""
This module contains the neural network models used in the CS172 assignment 1.
The model classes are subclasses of the torch.nn.Module class, which is a PyTorch class for creating neural network models.
Models are implemented using a combination of PyTorch's built-in layers and functions, such as nn.Linear, nn.Conv2d, and F.relu.
The forward method of the model class defines the forward pass of the neural network, which specifies how input data is processed to produce output predictions.
This module will be used in the training and evaluation of neural network models on various datasets.
"""

import torch
import torch.nn as nn
import torchvision


def get_model(model_name):
    if model_name == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = torch.nn.Linear(512, 50)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34()
        model.fc = torch.nn.Linear(512, 50)
    elif model_name == "myresnet18":
        # digit acc: 90.72%
        # image acc: 60.59%
        # epoch=10, transform=None
        model = ResNet18(3, 50)
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")
    return model


class SimpleResBlock(nn.Module):
    """
    A simple residual block for ResNet.
    The block consists of two convolutional layers with batch normalization and ReLU activation.
    The block also includes a skip connection to handle the case when the input and output dimensions do not match.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int): The stride for the convolutional layers.
        """
        super(SimpleResBlock, self).__init__()
        # ===================== TO DO Start =========================
        # You should set the args of each layer based on the implemented of residual block
        # self.dowmsample is needed to modify the channel num of residual
        # ===========================================================
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
        # ====================== TO DO END ==========================

    def forward(self, x):
        # ===================== TO DO Start =========================
        # The inputs x should be calculated sequentially with the variables defined in __init__
        # self.dowmsample is needed to modify the channel num of residual
        # ===========================================================
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
        # ====================== TO DO END ==========================
        return out


class ResNet18(nn.Module):
    """
    A simple implementation of the ResNet-18 architecture.
    The ResNet-18 architecture consists of a series of residual blocks with different numbers of layers.
    The architecture includes a convolutional layer, followed by four residual blocks, and a fully connected layer.
    """

    def __init__(self, in_channels, num_classes):
        """
        Args:
            in_channels (int): The number of input channels.
            num_classes (int): The number of classes in the dataset.
        """
        super(ResNet18, self).__init__()
        # ===================== TO DO Start =========================
        # You should set the args of each layer based on the implemented of resnet18
        # layer1/2/3/4 are residual blocks returned by self.__make_layer
        # ===========================================================
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
        self.fc = nn.Linear(512, num_classes)
        # ====================== TO DO END ==========================

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        # ===================== TO DO Start =========================
        # In this function, you should implement the residual block with SimpleResBlock
        # You may find nn.Sequential is a usefule function
        # ===========================================================
        layers = [SimpleResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(SimpleResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
        # ====================== TO DO END ==========================

    def forward(self, x):
        # ===================== TO DO Start =========================
        # The inputs x should be calculated sequentially with the variables defined in __init__
        # ===========================================================
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
        x = self.fc(x)
        # ====================== TO DO END ==========================
        return x
