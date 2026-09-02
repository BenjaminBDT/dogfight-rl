from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VisionHeadConfig:
    num_segmentation_classes: int = 3
    num_keypoints: int = 9
    embedding_dim: int = 128
    hidden_dim: int = 128


class SegmentationHead(nn.Module):
    def __init__(
        self,
        stage1_channels: int,
        stage2_channels: int,
        stage3_channels: int,
        final_channels: int,
        num_classes: int,
        *,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.stage1_proj = nn.Conv2d(stage1_channels, decoder_channels, kernel_size=1, bias=False)
        self.stage2_proj = nn.Conv2d(stage2_channels, decoder_channels, kernel_size=1, bias=False)
        self.stage3_proj = nn.Conv2d(stage3_channels, decoder_channels, kernel_size=1, bias=False)
        self.final_proj = nn.Conv2d(final_channels, decoder_channels, kernel_size=1, bias=False)
        self.stage3_refine = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.stage2_refine = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.stage1_refine = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(
        self,
        *,
        stage1: Tensor,
        stage2: Tensor,
        stage3: Tensor,
        final: Tensor,
        output_hw: tuple[int, int],
    ) -> Tensor:
        p4 = self.final_proj(final)
        p3 = self.stage3_proj(stage3) + nn.functional.interpolate(
            p4,
            size=stage3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p3 = self.stage3_refine(p3)
        p2 = self.stage2_proj(stage2) + nn.functional.interpolate(
            p3,
            size=stage2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p2 = self.stage2_refine(p2)
        p1 = self.stage1_proj(stage1) + nn.functional.interpolate(
            p2,
            size=stage1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        p1 = self.stage1_refine(p1)
        logits = self.classifier(p1)
        return nn.functional.interpolate(
            logits,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )


class KeypointHead(nn.Module):
    def __init__(self, in_channels: int, _hidden_dim: int, num_keypoints: int) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.voting_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, num_keypoints * 2, kernel_size=1),
        )

    def forward(
        self,
        features: Tensor,
        *,
        output_hw: tuple[int, int],
    ) -> Tensor:
        voting = self.voting_head(features)
        voting = nn.functional.interpolate(
            voting,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )
        voting = voting.reshape(
            features.shape[0],
            self.num_keypoints,
            2,
            output_hw[0],
            output_hw[1],
        )
        return voting


class VisualEmbeddingHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def pooled(self, features: Tensor) -> Tensor:
        return self.pool(features).flatten(1)

    def forward(self, selected_features: Tensor, selected_view_onehot: Tensor) -> Tensor:
        selected = self.pooled(selected_features)
        return self.proj(torch.cat([selected, selected_view_onehot], dim=1))
