from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

from .module import (
    SelectedVisualCandidateOutput,
    SingleStepVisionCandidateOutput,
    SingleStepVisionOutput,
)


@dataclass(frozen=True)
class DeepLabSingleStepVisionConfig:
    num_segmentation_classes: int
    embedding_dim: int = 128
    num_keypoints: int = 9
    pretrained: bool = True


class DeepLabSingleStepVisionModule(nn.Module):
    def __init__(self, config: DeepLabSingleStepVisionConfig) -> None:
        super().__init__()
        self.config = config
        backbone_weights = ResNet50_Weights.DEFAULT if config.pretrained else None
        self.segmentation_model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=config.num_segmentation_classes,
        )
        self.embedding_proj = nn.Linear(config.num_segmentation_classes * 2 + 3, config.embedding_dim)

    @staticmethod
    def _target_area(segmentation_logits: Tensor, target_class_ids: Tensor) -> Tensor:
        predicted = segmentation_logits.argmax(dim=1)
        target = target_class_ids.to(device=predicted.device, dtype=predicted.dtype).view(-1, 1, 1)
        return (predicted == target).sum(dim=(1, 2))

    @staticmethod
    def _select_view(front_area: Tensor, rear_area: Tensor) -> tuple[Tensor, Tensor]:
        selected_view_index = torch.full_like(front_area, 2, dtype=torch.long)
        front_mask = (front_area > rear_area) & (front_area > 0)
        rear_mask = (rear_area >= front_area) & (rear_area > 0)
        selected_view_index = torch.where(front_mask, torch.zeros_like(selected_view_index), selected_view_index)
        selected_view_index = torch.where(rear_mask, torch.ones_like(selected_view_index), selected_view_index)
        selected_view_onehot = F.one_hot(selected_view_index, num_classes=3).to(dtype=torch.float32)
        return selected_view_index, selected_view_onehot

    def _encode_view(self, image: Tensor) -> Tensor:
        return self.segmentation_model(image)["out"]

    def forward(
        self,
        front_image: Tensor,
        rear_image: Tensor,
        target_class_ids: Tensor | None = None,
    ) -> SingleStepVisionOutput:
        batch_size = front_image.shape[0]
        if target_class_ids is None:
            target_class_ids = torch.full((batch_size,), 2, dtype=torch.long, device=front_image.device)
        else:
            target_class_ids = target_class_ids.to(device=front_image.device, dtype=torch.long)

        front_segmentation_logits = self._encode_view(front_image)
        rear_segmentation_logits = self._encode_view(rear_image)

        front_pred_target_area = self._target_area(front_segmentation_logits, target_class_ids)
        rear_pred_target_area = self._target_area(rear_segmentation_logits, target_class_ids)
        selected_view_index, selected_view_onehot = self._select_view(front_pred_target_area, rear_pred_target_area)

        front_pooled = F.adaptive_avg_pool2d(front_segmentation_logits, (1, 1)).flatten(1)
        rear_pooled = F.adaptive_avg_pool2d(rear_segmentation_logits, (1, 1)).flatten(1)
        selected_logits = torch.zeros_like(front_pooled)
        front_mask = selected_view_index == 0
        rear_mask = selected_view_index == 1
        if front_mask.any():
            selected_logits[front_mask] = front_pooled[front_mask]
        if rear_mask.any():
            selected_logits[rear_mask] = rear_pooled[rear_mask]
        visual_embedding = self.embedding_proj(
            torch.cat([front_pooled, rear_pooled, selected_view_onehot.to(front_pooled.dtype)], dim=1)
        )

        num_keypoints = self.config.num_keypoints
        dtype = front_segmentation_logits.dtype
        device = front_segmentation_logits.device
        zeros_kp_xy = torch.zeros(batch_size, num_keypoints, 2, dtype=dtype, device=device)
        zeros_kp_support = torch.zeros(batch_size, num_keypoints, dtype=dtype, device=device)
        zeros_pnp = torch.zeros(batch_size, dtype=dtype, device=device)
        inf_reproj = torch.full((batch_size,), float("inf"), dtype=dtype, device=device)
        zeros_voting = torch.zeros(
            batch_size,
            num_keypoints,
            2,
            front_segmentation_logits.shape[-2],
            front_segmentation_logits.shape[-1],
            dtype=dtype,
            device=device,
        )
        zeros_evidence = torch.zeros(batch_size, dtype=dtype, device=device)
        zeros_pose = torch.zeros(batch_size, 6, dtype=dtype, device=device)
        front_candidate = SingleStepVisionCandidateOutput(
            visual_embedding=visual_embedding,
            segmentation_logits=front_segmentation_logits,
            voting_field=zeros_voting,
            keypoints_xy=zeros_kp_xy,
            keypoint_support=zeros_kp_support,
            pnp_success=zeros_pnp,
            reprojection_error=inf_reproj,
            v_sup=zeros_pnp,
            v_rep=zeros_pnp,
            raw_visual_evidence_strength=zeros_evidence,
            pred_target_area=front_pred_target_area,
            view_valid=(front_pred_target_area > 0).to(dtype=dtype),
            pos_valid=zeros_pnp,
            ori_valid=zeros_pnp,
            body_pose_9d=torch.zeros(batch_size, 9, dtype=dtype, device=device),
        )
        rear_candidate = SingleStepVisionCandidateOutput(
            visual_embedding=visual_embedding,
            segmentation_logits=rear_segmentation_logits,
            voting_field=zeros_voting.clone(),
            keypoints_xy=zeros_kp_xy.clone(),
            keypoint_support=zeros_kp_support.clone(),
            pnp_success=zeros_pnp.clone(),
            reprojection_error=inf_reproj.clone(),
            v_sup=zeros_pnp.clone(),
            v_rep=zeros_pnp.clone(),
            raw_visual_evidence_strength=zeros_evidence.clone(),
            pred_target_area=rear_pred_target_area,
            view_valid=(rear_pred_target_area > 0).to(dtype=dtype),
            pos_valid=zeros_pnp.clone(),
            ori_valid=zeros_pnp.clone(),
            body_pose_9d=torch.zeros(batch_size, 9, dtype=dtype, device=device),
        )
        selected_candidate = SelectedVisualCandidateOutput(
            view_index=selected_view_index,
            view_onehot=selected_view_onehot,
            view_changed=torch.zeros_like(selected_view_index, dtype=torch.bool),
            visual_embedding=visual_embedding,
            raw_visual_evidence_strength=zeros_evidence,
            keypoints_xy=zeros_kp_xy,
            keypoint_support=zeros_kp_support,
            pnp_success=zeros_pnp,
            reprojection_error=inf_reproj,
            v_sup=zeros_pnp,
            v_rep=zeros_pnp,
            pred_target_area=torch.where(
                selected_view_index == 0,
                front_pred_target_area,
                torch.where(selected_view_index == 1, rear_pred_target_area, torch.zeros_like(front_pred_target_area)),
            ),
            view_valid=(selected_view_index < 2).to(dtype=dtype),
            pos_valid=zeros_pnp,
            ori_valid=zeros_pnp,
            body_pose_9d=torch.zeros(batch_size, 9, dtype=dtype, device=device),
        )

        return SingleStepVisionOutput(
            front_candidate=front_candidate,
            rear_candidate=rear_candidate,
            selected_candidate=selected_candidate,
            front_segmentation_logits=front_segmentation_logits,
            rear_segmentation_logits=rear_segmentation_logits,
            front_keypoints_xy=zeros_kp_xy,
            rear_keypoints_xy=zeros_kp_xy.clone(),
            front_pnp_success=zeros_pnp,
            rear_pnp_success=zeros_pnp.clone(),
            front_reprojection_error=inf_reproj,
            rear_reprojection_error=inf_reproj.clone(),
            front_v_sup=zeros_pnp,
            rear_v_sup=zeros_pnp.clone(),
            front_v_rep=zeros_pnp,
            rear_v_rep=zeros_pnp.clone(),
            front_raw_visual_evidence_strength=zeros_evidence,
            rear_raw_visual_evidence_strength=zeros_evidence.clone(),
            raw_visual_evidence_strength=selected_candidate.raw_visual_evidence_strength,
            visual_embedding=selected_candidate.visual_embedding,
            front_voting_field=zeros_voting,
            rear_voting_field=zeros_voting.clone(),
            front_keypoint_support=zeros_kp_support,
            rear_keypoint_support=zeros_kp_support.clone(),
            selected_view_index=selected_view_index,
            selected_view_onehot=selected_view_onehot,
            selected_view_changed=selected_candidate.view_changed,
            front_pred_target_area=front_pred_target_area,
            rear_pred_target_area=rear_pred_target_area,
        )
