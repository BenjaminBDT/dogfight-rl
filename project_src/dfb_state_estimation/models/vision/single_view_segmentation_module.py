from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

from .backbone import VisionBackbone, VisionBackboneConfig
from .heads import SegmentationHead, VisionHeadConfig


@dataclass(frozen=True)
class SingleViewSegmentationOutput:
    segmentation_logits: Tensor
    pred_target_area: Tensor


@dataclass(frozen=True)
class SingleViewSegmentationConfig:
    backbone: VisionBackboneConfig = VisionBackboneConfig()
    heads: VisionHeadConfig = VisionHeadConfig()


@dataclass(frozen=True)
class DeepLabSingleViewSegmentationConfig:
    num_segmentation_classes: int
    pretrained: bool = True


def _target_area(segmentation_logits: Tensor, target_class_ids: Tensor) -> Tensor:
    predicted = segmentation_logits.argmax(dim=1)
    target = target_class_ids.to(device=predicted.device, dtype=predicted.dtype).view(-1, 1, 1)
    return (predicted == target).sum(dim=(1, 2))


class SingleViewSegmentationModule(nn.Module):
    def __init__(self, config: SingleViewSegmentationConfig | None = None) -> None:
        super().__init__()
        config = config or SingleViewSegmentationConfig()
        self.config = config
        self.backbone = VisionBackbone(config.backbone)
        self.segmentation_head = SegmentationHead(
            self.backbone.stage1_channels,
            self.backbone.stage2_channels,
            self.backbone.stage3_channels,
            self.backbone.out_channels,
            config.heads.num_segmentation_classes,
        )

    def forward(
        self,
        image: Tensor,
        target_class_ids: Tensor | None = None,
    ) -> SingleViewSegmentationOutput:
        batch_size = image.shape[0]
        if target_class_ids is None:
            target_class_ids = torch.full((batch_size,), 1, dtype=torch.long, device=image.device)
        else:
            target_class_ids = target_class_ids.to(device=image.device, dtype=torch.long)
        features = self.backbone(image)
        logits = self.segmentation_head(
            stage1=features.stage1,
            stage2=features.stage2,
            stage3=features.stage3,
            final=features.final,
            output_hw=(image.shape[-2], image.shape[-1]),
        )
        return SingleViewSegmentationOutput(
            segmentation_logits=logits,
            pred_target_area=_target_area(logits, target_class_ids),
        )


class DeepLabSingleViewSegmentationModule(nn.Module):
    def __init__(self, config: DeepLabSingleViewSegmentationConfig) -> None:
        super().__init__()
        self.config = config
        backbone_weights = ResNet50_Weights.DEFAULT if config.pretrained else None
        self.segmentation_model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=config.num_segmentation_classes,
        )

    def forward(
        self,
        image: Tensor,
        target_class_ids: Tensor | None = None,
    ) -> SingleViewSegmentationOutput:
        batch_size = image.shape[0]
        if target_class_ids is None:
            target_class_ids = torch.full((batch_size,), 1, dtype=torch.long, device=image.device)
        else:
            target_class_ids = target_class_ids.to(device=image.device, dtype=torch.long)
        logits = self.segmentation_model(image)["out"]
        return SingleViewSegmentationOutput(
            segmentation_logits=logits,
            pred_target_area=_target_area(logits, target_class_ids),
        )
