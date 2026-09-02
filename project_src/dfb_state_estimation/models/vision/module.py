from __future__ import annotations

from dataclasses import dataclass, field

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import VisionBackbone, VisionBackboneConfig
from .geometry import (
    GeometryValidationConfig,
    VisualGeometryValidator,
)
from .heads import KeypointHead, SegmentationHead, VisualEmbeddingHead, VisionHeadConfig


@dataclass(frozen=True)
class SingleStepVisionConfig:
    backbone: VisionBackboneConfig = VisionBackboneConfig()
    heads: VisionHeadConfig = VisionHeadConfig()
    aggregation: "VotingAggregationConfig" = field(default_factory=lambda: VotingAggregationConfig())
    geometry: GeometryValidationConfig = field(default_factory=GeometryValidationConfig)


@dataclass(frozen=True)
class VotingAggregationConfig:
    foreground_threshold: float = 0.5
    max_foreground_points: int = 256
    max_hypotheses: int = 64
    support_distance_sigma: float = 0.04
    support_softmax_temperature: float = 0.15


@dataclass(frozen=True)
class SingleStepVisionCandidateOutput:
    visual_embedding: Tensor
    segmentation_logits: Tensor
    voting_field: Tensor
    keypoints_xy: Tensor
    keypoint_support: Tensor
    pnp_success: Tensor
    reprojection_error: Tensor
    v_sup: Tensor
    v_rep: Tensor
    raw_visual_evidence_strength: Tensor
    pred_target_area: Tensor
    view_valid: Tensor
    pos_valid: Tensor
    ori_valid: Tensor
    body_pose_9d: Tensor


@dataclass(frozen=True)
class SelectedVisualCandidateOutput:
    view_index: Tensor
    view_onehot: Tensor
    view_changed: Tensor
    visual_embedding: Tensor
    raw_visual_evidence_strength: Tensor
    keypoints_xy: Tensor
    keypoint_support: Tensor
    pnp_success: Tensor
    reprojection_error: Tensor
    v_sup: Tensor
    v_rep: Tensor
    pred_target_area: Tensor
    view_valid: Tensor
    pos_valid: Tensor
    ori_valid: Tensor
    body_pose_9d: Tensor


@dataclass(frozen=True)
class SingleStepVisionOutput:
    front_candidate: SingleStepVisionCandidateOutput
    rear_candidate: SingleStepVisionCandidateOutput
    selected_candidate: SelectedVisualCandidateOutput
    front_segmentation_logits: Tensor
    rear_segmentation_logits: Tensor
    front_keypoints_xy: Tensor
    rear_keypoints_xy: Tensor
    front_pnp_success: Tensor
    rear_pnp_success: Tensor
    front_reprojection_error: Tensor
    rear_reprojection_error: Tensor
    front_v_sup: Tensor
    rear_v_sup: Tensor
    front_v_rep: Tensor
    rear_v_rep: Tensor
    front_raw_visual_evidence_strength: Tensor
    rear_raw_visual_evidence_strength: Tensor
    raw_visual_evidence_strength: Tensor
    visual_embedding: Tensor
    front_voting_field: Tensor
    rear_voting_field: Tensor
    front_keypoint_support: Tensor
    rear_keypoint_support: Tensor
    selected_view_index: Tensor
    selected_view_onehot: Tensor
    selected_view_changed: Tensor
    front_pred_target_area: Tensor
    rear_pred_target_area: Tensor


class SingleStepVisionModule(nn.Module):
    _FRONT_CAMERA_POSITION_BODY = (0.0, 10.0, -25.0)
    _REAR_CAMERA_POSITION_BODY = (0.0, 10.0, 25.0)

    def __init__(self, config: SingleStepVisionConfig | None = None) -> None:
        super().__init__()
        config = config or SingleStepVisionConfig()
        self.config = config
        self.backbone = VisionBackbone(config.backbone)
        self.segmentation_head = SegmentationHead(
            self.backbone.stage1_channels,
            self.backbone.stage2_channels,
            self.backbone.stage3_channels,
            self.backbone.out_channels,
            config.heads.num_segmentation_classes,
        )
        self.keypoint_head = KeypointHead(
            self.backbone.out_channels,
            config.heads.hidden_dim,
            config.heads.num_keypoints,
        )
        self.embedding_head = VisualEmbeddingHead(
            self.backbone.out_channels,
            config.heads.hidden_dim,
            config.heads.embedding_dim,
        )
        self.aggregation = config.aggregation
        self.geometry_validator = VisualGeometryValidator(config.geometry)

    @staticmethod
    def _foreground_probability(segmentation_logits: Tensor) -> Tensor:
        class_probs = torch.softmax(segmentation_logits, dim=1)
        return 1.0 - class_probs[:, :1]

    @staticmethod
    def _target_area(segmentation_logits: Tensor, target_class_ids: Tensor) -> Tensor:
        if segmentation_logits.ndim != 4:
            raise ValueError("segmentation_logits must be [B, C, H, W]")
        if target_class_ids.ndim != 1 or target_class_ids.shape[0] != segmentation_logits.shape[0]:
            raise ValueError("target_class_ids must be [B] and match segmentation batch dim")
        predicted = segmentation_logits.argmax(dim=1)
        target = target_class_ids.to(device=predicted.device, dtype=predicted.dtype).view(-1, 1, 1)
        return (predicted == target).sum(dim=(1, 2))

    @staticmethod
    def _routing_score(
        area: Tensor,
        *,
        pnp_success: Tensor,
        v_rep: Tensor,
        v_sup: Tensor,
    ) -> Tensor:
        return (
            1000.0 * pnp_success
            + 100.0 * v_rep
            + 10.0 * v_sup
            + 0.001 * torch.log1p(area.to(dtype=torch.float32))
        )

    @classmethod
    def _select_view(
        cls,
        front_area: Tensor,
        rear_area: Tensor,
        *,
        front_pnp_success: Tensor,
        rear_pnp_success: Tensor,
        front_v_rep: Tensor,
        rear_v_rep: Tensor,
        front_v_sup: Tensor,
        rear_v_sup: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if front_area.shape != rear_area.shape:
            raise ValueError("front_area and rear_area must share shape")
        front_score = cls._routing_score(
            front_area,
            pnp_success=front_pnp_success,
            v_rep=front_v_rep,
            v_sup=front_v_sup,
        )
        rear_score = cls._routing_score(
            rear_area,
            pnp_success=rear_pnp_success,
            v_rep=rear_v_rep,
            v_sup=rear_v_sup,
        )
        front_valid = (front_area > 0) | (front_v_sup > 0.0) | (front_pnp_success > 0.0)
        rear_valid = (rear_area > 0) | (rear_v_sup > 0.0) | (rear_pnp_success > 0.0)
        selected_view_index = torch.full_like(front_area, 2, dtype=torch.long)
        front_mask = front_valid & (~rear_valid | (front_score > rear_score))
        rear_mask = rear_valid & (~front_valid | ~front_mask)
        selected_view_index = torch.where(
            front_mask,
            torch.zeros_like(selected_view_index),
            selected_view_index,
        )
        selected_view_index = torch.where(
            rear_mask,
            torch.ones_like(selected_view_index),
            selected_view_index,
        )
        selected_view_onehot = F.one_hot(selected_view_index, num_classes=3).to(
            dtype=torch.float32
        )
        return selected_view_index, selected_view_onehot

    @staticmethod
    def _select_tensor(front_value: Tensor, rear_value: Tensor, selected_view_index: Tensor) -> Tensor:
        if front_value.shape != rear_value.shape:
            raise ValueError("front_value and rear_value must share shape")
        if front_value.shape[0] != selected_view_index.shape[0]:
            raise ValueError("selected_view_index batch dim must match front/rear values")
        selected = torch.zeros_like(front_value)
        front_mask = selected_view_index == 0
        rear_mask = selected_view_index == 1
        if front_mask.any():
            selected[front_mask] = front_value[front_mask]
        if rear_mask.any():
            selected[rear_mask] = rear_value[rear_mask]
        return selected

    @staticmethod
    def _select_feature_map(
        front_features: Tensor,
        rear_features: Tensor,
        selected_view_index: Tensor,
    ) -> Tensor:
        if front_features.shape != rear_features.shape:
            raise ValueError("front_features and rear_features must share shape")
        if front_features.shape[0] != selected_view_index.shape[0]:
            raise ValueError("selected_view_index batch dim must match feature maps")
        selected = torch.zeros_like(front_features)
        front_mask = selected_view_index == 0
        rear_mask = selected_view_index == 1
        if front_mask.any():
            selected[front_mask] = front_features[front_mask]
        if rear_mask.any():
            selected[rear_mask] = rear_features[rear_mask]
        return selected

    @staticmethod
    def _candidate_view_valid(
        *,
        pred_target_area: Tensor,
        v_sup: Tensor,
        pnp_success: Tensor,
    ) -> Tensor:
        valid = (pred_target_area > 0) | (v_sup > 0.0) | (pnp_success > 0.0)
        return valid.to(dtype=v_sup.dtype)

    @staticmethod
    def _candidate_pose_valid(pnp_success: Tensor) -> Tensor:
        return (pnp_success > 0.5).to(dtype=pnp_success.dtype)

    def _body_pose_valid(self, body_pose_9d: Tensor, pnp_success: Tensor) -> Tensor:
        if body_pose_9d.ndim != 2 or body_pose_9d.shape[-1] != 9:
            raise ValueError("body_pose_9d must be [B, 9]")
        translation = body_pose_9d[:, :3]
        translation_norm = torch.linalg.vector_norm(translation, dim=-1)
        sane = torch.isfinite(body_pose_9d).all(dim=-1)
        sane = sane & (
            translation_norm
            <= float(self.config.geometry.pnp_success_max_camera_translation_norm)
        )
        return (pnp_success > 0.5).to(dtype=body_pose_9d.dtype) * sane.to(dtype=body_pose_9d.dtype)

    @staticmethod
    def _view_onehot(
        *,
        batch_size: int,
        view_index: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        index = torch.full((batch_size,), view_index, dtype=torch.long, device=device)
        return F.one_hot(index, num_classes=3).to(dtype=dtype)

    @staticmethod
    def _rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
        a1 = rotation_6d[..., 0:3]
        a2 = rotation_6d[..., 3:6]
        b1 = F.normalize(a1, dim=-1)
        proj = (b1 * a2).sum(dim=-1, keepdim=True)
        b2 = F.normalize(a2 - proj * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    @staticmethod
    def _matrix_to_rotation_6d(rotation_matrix: Tensor) -> Tensor:
        return torch.cat([rotation_matrix[..., :, 0], rotation_matrix[..., :, 1]], dim=-1)

    @staticmethod
    def _axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
        if axis_angle.shape[-1] != 3:
            raise ValueError("axis_angle must end with 3 dims")
        theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
        k = axis_angle / theta.clamp_min(1.0e-6)
        kx, ky, kz = k.unbind(dim=-1)
        zeros = torch.zeros_like(kx)
        skew = torch.stack(
            [
                torch.stack([zeros, -kz, ky], dim=-1),
                torch.stack([kz, zeros, -kx], dim=-1),
                torch.stack([-ky, kx, zeros], dim=-1),
            ],
            dim=-2,
        )
        eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(
            axis_angle.shape[:-1] + (3, 3)
        )
        sin_theta = torch.sin(theta)[..., None]
        cos_theta = torch.cos(theta)[..., None]
        rotation = eye + sin_theta * skew + (1.0 - cos_theta) * torch.matmul(skew, skew)
        small = theta.squeeze(-1) < 1.0e-6
        if small.any():
            rotation = torch.where(
                small[..., None, None],
                eye + SingleStepVisionModule._skew_from_vector(axis_angle),
                rotation,
            )
        return rotation

    @staticmethod
    def _skew_from_vector(vector: Tensor) -> Tensor:
        vx, vy, vz = vector.unbind(dim=-1)
        zeros = torch.zeros_like(vx)
        return torch.stack(
            [
                torch.stack([zeros, -vz, vy], dim=-1),
                torch.stack([vz, zeros, -vx], dim=-1),
                torch.stack([-vy, vx, zeros], dim=-1),
            ],
            dim=-2,
        )

    @classmethod
    def _camera_extrinsics_body(
        cls,
        *,
        batch_size: int,
        view_is_front: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        position = cls._FRONT_CAMERA_POSITION_BODY if view_is_front else cls._REAR_CAMERA_POSITION_BODY
        if view_is_front:
            rotation = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=dtype,
                device=device,
            )
        else:
            rotation = torch.eye(3, dtype=dtype, device=device)
        position_tensor = torch.tensor(position, dtype=dtype, device=device).expand(batch_size, 3)
        rotation_tensor = rotation.expand(batch_size, 3, 3)
        return position_tensor, rotation_tensor

    @classmethod
    def _camera_pose_to_body_pose(
        cls,
        camera_pose_6d: Tensor,
        *,
        pnp_success: Tensor,
        view_is_front: bool,
    ) -> Tensor:
        if camera_pose_6d.ndim != 2 or camera_pose_6d.shape[-1] != 6:
            raise ValueError("camera_pose_6d must be [B, 6]")
        if pnp_success.shape != (camera_pose_6d.shape[0],):
            raise ValueError("pnp_success must be [B]")
        batch_size = camera_pose_6d.shape[0]
        t_bc, r_bc = cls._camera_extrinsics_body(
            batch_size=batch_size,
            view_is_front=view_is_front,
            dtype=camera_pose_6d.dtype,
            device=camera_pose_6d.device,
        )
        r_co = cls._axis_angle_to_matrix(camera_pose_6d[:, :3])
        t_co = camera_pose_6d[:, 3:]
        r_bo = torch.matmul(r_bc, r_co)
        t_bo = torch.matmul(r_bc, t_co.unsqueeze(-1)).squeeze(-1) + t_bc
        pose = torch.cat([t_bo, cls._matrix_to_rotation_6d(r_bo)], dim=-1)
        valid = (pnp_success > 0.5).unsqueeze(-1).to(dtype=pose.dtype)
        return pose * valid

    @classmethod
    def _route_candidates(
        cls,
        *,
        front_candidate: SingleStepVisionCandidateOutput,
        rear_candidate: SingleStepVisionCandidateOutput,
    ) -> tuple[Tensor, Tensor]:
        selected_view_index, selected_view_onehot = cls._select_view(
            front_candidate.pred_target_area,
            rear_candidate.pred_target_area,
            front_pnp_success=front_candidate.pnp_success,
            rear_pnp_success=rear_candidate.pnp_success,
            front_v_rep=front_candidate.v_rep,
            rear_v_rep=rear_candidate.v_rep,
            front_v_sup=front_candidate.v_sup,
            rear_v_sup=rear_candidate.v_sup,
        )
        return selected_view_index, selected_view_onehot

    @staticmethod
    def _pixel_grid(
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        ys = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _aggregate_keypoints_from_voting(
        self,
        voting_field: Tensor,
        segmentation_logits: Tensor,
        *,
        foreground_override: Tensor | None = None,
        foreground_override_mix: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        batch_size, num_keypoints, _, height, width = voting_field.shape
        dtype = voting_field.dtype
        device = voting_field.device
        pred_foreground = self._foreground_probability(segmentation_logits).clamp_min(1.0e-6)
        if foreground_override is not None:
            if foreground_override.shape != (batch_size, 1, height, width):
                raise ValueError("foreground_override must be [B, 1, H, W]")
            mix = float(min(max(foreground_override_mix, 0.0), 1.0))
            gt_foreground = foreground_override.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            foreground = (mix * gt_foreground + (1.0 - mix) * pred_foreground).clamp_min(1.0e-6)
        else:
            foreground = pred_foreground
        pixel_grid = self._pixel_grid(
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        directions = F.normalize(voting_field, dim=2, eps=1.0e-6)
        directions_xy = directions.permute(0, 1, 3, 4, 2).reshape(
            batch_size, num_keypoints, height * width, 2
        )
        pixels_xy = pixel_grid.permute(0, 2, 3, 1).reshape(batch_size, height * width, 2)
        foreground_flat = foreground.reshape(batch_size, height * width)
        keypoints_xy = torch.zeros(batch_size, num_keypoints, 2, device=device, dtype=dtype)
        keypoint_support = torch.zeros(batch_size, num_keypoints, device=device, dtype=dtype)
        all_indices = torch.arange(height * width, device=device)

        for batch_index in range(batch_size):
            fg_weights = foreground_flat[batch_index]
            fg_mask = fg_weights >= self.aggregation.foreground_threshold
            selected = all_indices[fg_mask]
            if selected.numel() < 2:
                selected = torch.topk(
                    fg_weights,
                    k=min(self.aggregation.max_foreground_points, fg_weights.numel()),
                ).indices
            if selected.numel() > self.aggregation.max_foreground_points:
                selected = torch.topk(
                    fg_weights[selected],
                    k=self.aggregation.max_foreground_points,
                ).indices.to(device=selected.device)
                selected = all_indices[fg_mask][selected]

            if selected.numel() < 2:
                continue

            selected_pixels = pixels_xy[batch_index, selected]
            selected_weights = fg_weights[selected]
            pair_count = min(
                self.aggregation.max_hypotheses,
                max((selected.numel() * (selected.numel() - 1)) // 2, 1),
            )
            pair_a: list[int] = []
            pair_b: list[int] = []
            cursor = 0
            for left in range(selected.numel() - 1):
                for right in range(left + 1, selected.numel()):
                    pair_a.append(left)
                    pair_b.append(right)
                    cursor += 1
                    if cursor >= pair_count:
                        break
                if cursor >= pair_count:
                    break
            pair_a_idx = torch.tensor(pair_a, device=device, dtype=torch.long)
            pair_b_idx = torch.tensor(pair_b, device=device, dtype=torch.long)

            for keypoint_index in range(num_keypoints):
                dirs = directions_xy[batch_index, keypoint_index, selected]
                p1 = selected_pixels[pair_a_idx]
                p2 = selected_pixels[pair_b_idx]
                d1 = dirs[pair_a_idx]
                d2 = dirs[pair_b_idx]
                system = torch.stack([d1, -d2], dim=-1)
                rhs = (p2 - p1).unsqueeze(-1)
                det = system[:, 0, 0] * system[:, 1, 1] - system[:, 0, 1] * system[:, 1, 0]
                valid_pairs = det.abs() > 1.0e-4
                if not bool(valid_pairs.any()):
                    hypothesis = selected_pixels.mean(dim=0)
                    keypoints_xy[batch_index, keypoint_index] = hypothesis.clamp(0.0, 1.0)
                    keypoint_support[batch_index, keypoint_index] = 0.0
                    continue

                safe_system = system[valid_pairs]
                safe_rhs = rhs[valid_pairs]
                safe_p1 = p1[valid_pairs]
                t_values = torch.linalg.solve(safe_system, safe_rhs).squeeze(-1)[:, 0:1]
                hypotheses = (safe_p1 + t_values * d1[valid_pairs]).clamp(0.0, 1.0)

                diffs = hypotheses.unsqueeze(1) - selected_pixels.unsqueeze(0)
                dir_all = dirs.unsqueeze(0)
                parallel = (diffs * dir_all).sum(dim=-1, keepdim=True) * dir_all
                perp = diffs - parallel
                distances = torch.linalg.norm(perp, dim=-1)
                sigma = self.aggregation.support_distance_sigma
                support_scores = (
                    selected_weights.unsqueeze(0)
                    * torch.exp(-distances.pow(2) / max(2.0 * sigma * sigma, 1.0e-6))
                ).sum(dim=1)
                temperature = max(self.aggregation.support_softmax_temperature, 1.0e-3)
                hypothesis_weights = torch.softmax(support_scores / temperature, dim=0)
                hypothesis = (hypothesis_weights.unsqueeze(-1) * hypotheses).sum(dim=0)
                support = (
                    support_scores.max() / selected_weights.sum().clamp_min(1.0e-6)
                ).clamp(0.0, 1.0)
                keypoints_xy[batch_index, keypoint_index] = hypothesis.clamp(0.0, 1.0)
                keypoint_support[batch_index, keypoint_index] = support

        return keypoints_xy, keypoint_support

    def forward(
        self,
        front_image: Tensor,
        rear_image: Tensor,
        target_class_ids: Tensor | None = None,
        front_aggregation_foreground: Tensor | None = None,
        rear_aggregation_foreground: Tensor | None = None,
        aggregation_foreground_mix: float = 1.0,
    ) -> SingleStepVisionOutput:
        if front_image.ndim != 4 or rear_image.ndim != 4:
            raise ValueError("front_image and rear_image must be [B, C, H, W] tensors")
        if front_image.shape[-2:] != rear_image.shape[-2:]:
            raise ValueError("front and rear image tensors must share spatial resolution")
        batch_size = front_image.shape[0]
        if target_class_ids is None:
            target_class_ids = torch.full(
                (batch_size,),
                2,
                dtype=torch.long,
                device=front_image.device,
            )
        elif target_class_ids.ndim != 1 or target_class_ids.shape[0] != batch_size:
            raise ValueError("target_class_ids must be [B]")
        else:
            target_class_ids = target_class_ids.to(device=front_image.device, dtype=torch.long)

        output_hw = (front_image.shape[-2], front_image.shape[-1])
        front_features = self.backbone(front_image)
        rear_features = self.backbone(rear_image)

        front_segmentation_logits = self.segmentation_head(
            stage1=front_features.stage1,
            stage2=front_features.stage2,
            stage3=front_features.stage3,
            final=front_features.final,
            output_hw=output_hw,
        )
        rear_segmentation_logits = self.segmentation_head(
            stage1=rear_features.stage1,
            stage2=rear_features.stage2,
            stage3=rear_features.stage3,
            final=rear_features.final,
            output_hw=output_hw,
        )
        front_voting_field = self.keypoint_head(
            front_features.final,
            output_hw=output_hw,
        )
        rear_voting_field = self.keypoint_head(
            rear_features.final,
            output_hw=output_hw,
        )
        front_keypoints_xy, front_keypoint_support = self._aggregate_keypoints_from_voting(
            front_voting_field,
            front_segmentation_logits,
            foreground_override=front_aggregation_foreground,
            foreground_override_mix=aggregation_foreground_mix,
        )
        rear_keypoints_xy, rear_keypoint_support = self._aggregate_keypoints_from_voting(
            rear_voting_field,
            rear_segmentation_logits,
            foreground_override=rear_aggregation_foreground,
            foreground_override_mix=aggregation_foreground_mix,
        )
        front_geometry = self.geometry_validator.evaluate_batch(
            front_keypoints_xy,
            front_keypoint_support,
            image_height=output_hw[0],
            image_width=output_hw[1],
        )
        rear_geometry = self.geometry_validator.evaluate_batch(
            rear_keypoints_xy,
            rear_keypoint_support,
            image_height=output_hw[0],
            image_width=output_hw[1],
        )
        front_pred_target_area = self._target_area(front_segmentation_logits, target_class_ids)
        rear_pred_target_area = self._target_area(rear_segmentation_logits, target_class_ids)
        front_visual_embedding = self.embedding_head(
            front_features.final,
            self._view_onehot(
                batch_size=batch_size,
                view_index=0,
                dtype=front_features.final.dtype,
                device=front_image.device,
            ),
        )
        rear_visual_embedding = self.embedding_head(
            rear_features.final,
            self._view_onehot(
                batch_size=batch_size,
                view_index=1,
                dtype=rear_features.final.dtype,
                device=front_image.device,
            ),
        )
        front_candidate = SingleStepVisionCandidateOutput(
            visual_embedding=front_visual_embedding,
            segmentation_logits=front_segmentation_logits,
            voting_field=front_voting_field,
            keypoints_xy=front_keypoints_xy,
            keypoint_support=front_keypoint_support,
            pnp_success=front_geometry.pnp_success,
            reprojection_error=front_geometry.reprojection_error,
            v_sup=front_geometry.v_sup,
            v_rep=front_geometry.v_rep,
            raw_visual_evidence_strength=front_geometry.raw_visual_evidence_strength,
            pred_target_area=front_pred_target_area,
            view_valid=self._candidate_view_valid(
                pred_target_area=front_pred_target_area,
                v_sup=front_geometry.v_sup,
                pnp_success=front_geometry.pnp_success,
            ),
            pos_valid=self._candidate_pose_valid(front_geometry.pnp_success),
            ori_valid=self._candidate_pose_valid(front_geometry.pnp_success),
            body_pose_9d=self._camera_pose_to_body_pose(
                front_geometry.camera_pose_6d,
                pnp_success=front_geometry.pnp_success,
                view_is_front=True,
            ),
        )
        front_body_pose_valid = self._body_pose_valid(
            front_candidate.body_pose_9d,
            front_candidate.pnp_success,
        )
        front_candidate = SingleStepVisionCandidateOutput(
            visual_embedding=front_candidate.visual_embedding,
            segmentation_logits=front_candidate.segmentation_logits,
            voting_field=front_candidate.voting_field,
            keypoints_xy=front_candidate.keypoints_xy,
            keypoint_support=front_candidate.keypoint_support,
            pnp_success=front_body_pose_valid,
            reprojection_error=front_candidate.reprojection_error,
            v_sup=front_candidate.v_sup,
            v_rep=front_candidate.v_rep * front_body_pose_valid,
            raw_visual_evidence_strength=0.5 * front_candidate.v_sup + 0.5 * (front_candidate.v_rep * front_body_pose_valid),
            pred_target_area=front_candidate.pred_target_area,
            view_valid=self._candidate_view_valid(
                pred_target_area=front_candidate.pred_target_area,
                v_sup=front_candidate.v_sup,
                pnp_success=front_body_pose_valid,
            ),
            pos_valid=front_body_pose_valid,
            ori_valid=front_body_pose_valid,
            body_pose_9d=front_candidate.body_pose_9d * front_body_pose_valid.unsqueeze(-1),
        )
        rear_candidate = SingleStepVisionCandidateOutput(
            visual_embedding=rear_visual_embedding,
            segmentation_logits=rear_segmentation_logits,
            voting_field=rear_voting_field,
            keypoints_xy=rear_keypoints_xy,
            keypoint_support=rear_keypoint_support,
            pnp_success=rear_geometry.pnp_success,
            reprojection_error=rear_geometry.reprojection_error,
            v_sup=rear_geometry.v_sup,
            v_rep=rear_geometry.v_rep,
            raw_visual_evidence_strength=rear_geometry.raw_visual_evidence_strength,
            pred_target_area=rear_pred_target_area,
            view_valid=self._candidate_view_valid(
                pred_target_area=rear_pred_target_area,
                v_sup=rear_geometry.v_sup,
                pnp_success=rear_geometry.pnp_success,
            ),
            pos_valid=self._candidate_pose_valid(rear_geometry.pnp_success),
            ori_valid=self._candidate_pose_valid(rear_geometry.pnp_success),
            body_pose_9d=self._camera_pose_to_body_pose(
                rear_geometry.camera_pose_6d,
                pnp_success=rear_geometry.pnp_success,
                view_is_front=False,
            ),
        )
        rear_body_pose_valid = self._body_pose_valid(
            rear_candidate.body_pose_9d,
            rear_candidate.pnp_success,
        )
        rear_candidate = SingleStepVisionCandidateOutput(
            visual_embedding=rear_candidate.visual_embedding,
            segmentation_logits=rear_candidate.segmentation_logits,
            voting_field=rear_candidate.voting_field,
            keypoints_xy=rear_candidate.keypoints_xy,
            keypoint_support=rear_candidate.keypoint_support,
            pnp_success=rear_body_pose_valid,
            reprojection_error=rear_candidate.reprojection_error,
            v_sup=rear_candidate.v_sup,
            v_rep=rear_candidate.v_rep * rear_body_pose_valid,
            raw_visual_evidence_strength=0.5 * rear_candidate.v_sup + 0.5 * (rear_candidate.v_rep * rear_body_pose_valid),
            pred_target_area=rear_candidate.pred_target_area,
            view_valid=self._candidate_view_valid(
                pred_target_area=rear_candidate.pred_target_area,
                v_sup=rear_candidate.v_sup,
                pnp_success=rear_body_pose_valid,
            ),
            pos_valid=rear_body_pose_valid,
            ori_valid=rear_body_pose_valid,
            body_pose_9d=rear_candidate.body_pose_9d * rear_body_pose_valid.unsqueeze(-1),
        )
        selected_view_index, selected_view_onehot = self._route_candidates(
            front_candidate=front_candidate,
            rear_candidate=rear_candidate,
        )
        visual_embedding = self._select_tensor(
            front_candidate.visual_embedding,
            rear_candidate.visual_embedding,
            selected_view_index,
        )
        selected_candidate = SelectedVisualCandidateOutput(
            view_index=selected_view_index,
            view_onehot=selected_view_onehot,
            view_changed=torch.zeros_like(selected_view_index, dtype=torch.bool),
            visual_embedding=visual_embedding,
            raw_visual_evidence_strength=self._select_tensor(
                front_candidate.raw_visual_evidence_strength,
                rear_candidate.raw_visual_evidence_strength,
                selected_view_index,
            ),
            keypoints_xy=self._select_tensor(
                front_candidate.keypoints_xy,
                rear_candidate.keypoints_xy,
                selected_view_index,
            ),
            keypoint_support=self._select_tensor(
                front_candidate.keypoint_support,
                rear_candidate.keypoint_support,
                selected_view_index,
            ),
            pnp_success=self._select_tensor(
                front_candidate.pnp_success,
                rear_candidate.pnp_success,
                selected_view_index,
            ),
            reprojection_error=self._select_tensor(
                front_candidate.reprojection_error,
                rear_candidate.reprojection_error,
                selected_view_index,
            ),
            v_sup=self._select_tensor(
                front_candidate.v_sup,
                rear_candidate.v_sup,
                selected_view_index,
            ),
            v_rep=self._select_tensor(
                front_candidate.v_rep,
                rear_candidate.v_rep,
                selected_view_index,
            ),
            pred_target_area=self._select_tensor(
                front_candidate.pred_target_area,
                rear_candidate.pred_target_area,
                selected_view_index,
            ),
            view_valid=(selected_view_index < 2).to(dtype=front_candidate.view_valid.dtype),
            pos_valid=self._select_tensor(
                front_candidate.pos_valid,
                rear_candidate.pos_valid,
                selected_view_index,
            ),
            ori_valid=self._select_tensor(
                front_candidate.ori_valid,
                rear_candidate.ori_valid,
                selected_view_index,
            ),
            body_pose_9d=self._select_tensor(
                front_candidate.body_pose_9d,
                rear_candidate.body_pose_9d,
                selected_view_index,
            ),
        )

        return SingleStepVisionOutput(
            front_candidate=front_candidate,
            rear_candidate=rear_candidate,
            selected_candidate=selected_candidate,
            front_segmentation_logits=front_segmentation_logits,
            rear_segmentation_logits=rear_segmentation_logits,
            front_keypoints_xy=front_keypoints_xy,
            rear_keypoints_xy=rear_keypoints_xy,
            front_pnp_success=front_geometry.pnp_success,
            rear_pnp_success=rear_geometry.pnp_success,
            front_reprojection_error=front_geometry.reprojection_error,
            rear_reprojection_error=rear_geometry.reprojection_error,
            front_v_sup=front_geometry.v_sup,
            rear_v_sup=rear_geometry.v_sup,
            front_v_rep=front_geometry.v_rep,
            rear_v_rep=rear_geometry.v_rep,
            front_raw_visual_evidence_strength=front_geometry.raw_visual_evidence_strength,
            rear_raw_visual_evidence_strength=rear_geometry.raw_visual_evidence_strength,
            raw_visual_evidence_strength=selected_candidate.raw_visual_evidence_strength,
            visual_embedding=selected_candidate.visual_embedding,
            front_voting_field=front_voting_field,
            rear_voting_field=rear_voting_field,
            front_keypoint_support=front_keypoint_support,
            rear_keypoint_support=rear_keypoint_support,
            selected_view_index=selected_view_index,
            selected_view_onehot=selected_view_onehot,
            selected_view_changed=selected_candidate.view_changed,
            front_pred_target_area=front_pred_target_area,
            rear_pred_target_area=rear_pred_target_area,
        )


def test_geometry_routing_prefers_successful_view_over_larger_area() -> None:
    front_area = torch.tensor([8000], dtype=torch.long)
    rear_area = torch.tensor([400], dtype=torch.long)
    selected_view_index, _ = SingleStepVisionModule._select_view(
        front_area,
        rear_area,
        front_pnp_success=torch.tensor([0.0]),
        rear_pnp_success=torch.tensor([1.0]),
        front_v_rep=torch.tensor([0.0]),
        rear_v_rep=torch.tensor([0.8]),
        front_v_sup=torch.tensor([0.9]),
        rear_v_sup=torch.tensor([0.99]),
    )
    assert selected_view_index.tolist() == [1]


def test_geometry_routing_falls_back_to_support_when_both_fail_pnp() -> None:
    front_area = torch.tensor([100], dtype=torch.long)
    rear_area = torch.tensor([5000], dtype=torch.long)
    selected_view_index, _ = SingleStepVisionModule._select_view(
        front_area,
        rear_area,
        front_pnp_success=torch.tensor([0.0]),
        rear_pnp_success=torch.tensor([0.0]),
        front_v_rep=torch.tensor([0.6]),
        rear_v_rep=torch.tensor([0.0]),
        front_v_sup=torch.tensor([0.95]),
        rear_v_sup=torch.tensor([0.6]),
    )
    assert selected_view_index.tolist() == [0]
