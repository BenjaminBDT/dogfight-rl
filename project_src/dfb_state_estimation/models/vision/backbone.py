from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class VisionBackboneConfig:
    name: str = "conv"
    pretrained: bool = False
    in_channels: int = 3
    stem_channels: int = 32
    hidden_channels: tuple[int, int, int] = (32, 64, 128)


@dataclass(frozen=True)
class VisionBackboneOutput:
    stage1: Tensor
    stage2: Tensor
    stage3: Tensor
    final: Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class VisionBackbone(nn.Module):
    def __init__(self, config: VisionBackboneConfig) -> None:
        super().__init__()
        self.name = config.name
        if config.name == "conv":
            c1, c2, c3 = config.hidden_channels
            self.stem = nn.Sequential(
                nn.Conv2d(
                    config.in_channels,
                    config.stem_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(config.stem_channels),
                nn.GELU(),
            )
            self.stage1 = ConvBlock(config.stem_channels, c1, stride=1)
            self.stage2 = ConvBlock(c1, c2, stride=2)
            self.stage3 = ConvBlock(c2, c3, stride=2)
            self.stage1_channels = c1
            self.stage2_channels = c2
            self.stage3_channels = c3
            self.out_channels = c3
            return
        if config.name == "resnet18":
            if config.in_channels != 3:
                raise ValueError("resnet18 backbone requires in_channels=3")
            weights = ResNet18_Weights.DEFAULT if config.pretrained else None
            backbone = resnet18(weights=weights)
            self.stem = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
            )
            self.stage1 = backbone.layer1
            self.stage2 = backbone.layer2
            self.stage3 = backbone.layer3
            self.stage4 = backbone.layer4
            self.stage1_channels = 64
            self.stage2_channels = 128
            self.stage3_channels = 256
            self.out_channels = 512
            return
        raise ValueError(f"unsupported vision backbone: {config.name}")

    def forward(self, x: Tensor) -> VisionBackboneOutput:
        x = self.stem(x)
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        if self.name == "resnet18":
            final = self.stage4(stage3)
        else:
            final = stage3
        return VisionBackboneOutput(
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            final=final,
        )
