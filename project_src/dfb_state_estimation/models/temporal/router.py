from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from dfb_state_estimation.models.vision.module import (
    SelectedVisualCandidateOutput,
    SingleStepVisionOutput,
)


@dataclass(frozen=True)
class TemporalVisualRoutingOutput:
    selected_view_index_t: Tensor
    selected_view_changed_t: Tensor


def _rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - proj * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _rotation_angle_between_6d(first: Tensor, second: Tensor) -> Tensor:
    first_r = _rotation_6d_to_matrix(first)
    second_r = _rotation_6d_to_matrix(second)
    rel = torch.matmul(first_r.transpose(-1, -2), second_r)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1.0e-6)
    return torch.acos(cos_theta)


def _geometry_routing_score(
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


def route_selected_view_with_inertia_t(
    *,
    front_pnp_success_t: Tensor,
    rear_pnp_success_t: Tensor,
    front_v_sup_t: Tensor,
    rear_v_sup_t: Tensor,
    front_v_rep_t: Tensor,
    rear_v_rep_t: Tensor,
    front_pred_target_area_t: Tensor,
    rear_pred_target_area_t: Tensor,
    front_body_pose_9d_t: Tensor,
    rear_body_pose_9d_t: Tensor,
) -> TemporalVisualRoutingOutput:
    if front_pnp_success_t.shape != rear_pnp_success_t.shape:
        raise ValueError("front/rear pnp success tensors must match")
    if front_body_pose_9d_t.shape != rear_body_pose_9d_t.shape:
        raise ValueError("front/rear body pose tensors must match")
    if front_body_pose_9d_t.ndim != 3 or front_body_pose_9d_t.shape[-1] != 9:
        raise ValueError("body pose tensors must be [B, T, 9]")
    batch_size, steps = front_pnp_success_t.shape
    selected = torch.full_like(front_pnp_success_t, 2, dtype=torch.long)
    changed = torch.zeros_like(front_pnp_success_t, dtype=torch.bool)
    if steps == 0:
        return TemporalVisualRoutingOutput(selected_view_index_t=selected, selected_view_changed_t=changed)

    front_score = _geometry_routing_score(
        front_pred_target_area_t,
        pnp_success=front_pnp_success_t,
        v_rep=front_v_rep_t,
        v_sup=front_v_sup_t,
    )
    rear_score = _geometry_routing_score(
        rear_pred_target_area_t,
        pnp_success=rear_pnp_success_t,
        v_rep=rear_v_rep_t,
        v_sup=rear_v_sup_t,
    )
    front_valid = (front_pred_target_area_t > 0) | (front_v_sup_t > 0.0) | (front_pnp_success_t > 0.0)
    rear_valid = (rear_pred_target_area_t > 0) | (rear_v_sup_t > 0.0) | (rear_pnp_success_t > 0.0)

    first_front = front_valid[:, 0] & (~rear_valid[:, 0] | (front_score[:, 0] >= rear_score[:, 0]))
    first_rear = rear_valid[:, 0] & (~front_valid[:, 0] | ~first_front)
    selected[:, 0] = torch.where(first_front, torch.zeros_like(selected[:, 0]), selected[:, 0])
    selected[:, 0] = torch.where(first_rear, torch.ones_like(selected[:, 0]), selected[:, 0])

    for step_index in range(1, steps):
        prev_view = selected[:, step_index - 1]
        prev_front = front_body_pose_9d_t[:, step_index - 1]
        prev_rear = rear_body_pose_9d_t[:, step_index - 1]
        prev_pose = torch.zeros_like(prev_front)
        front_prev_mask = prev_view == 0
        rear_prev_mask = prev_view == 1
        if front_prev_mask.any():
            prev_pose[front_prev_mask] = prev_front[front_prev_mask]
        if rear_prev_mask.any():
            prev_pose[rear_prev_mask] = prev_rear[rear_prev_mask]
        prev_valid = prev_view < 2

        front_success = front_pnp_success_t[:, step_index] > 0.5
        rear_success = rear_pnp_success_t[:, step_index] > 0.5
        front_only = front_success & ~rear_success
        rear_only = rear_success & ~front_success

        current = selected[:, step_index]
        current = torch.where(front_only, torch.zeros_like(current), current)
        current = torch.where(rear_only, torch.ones_like(current), current)

        both_success = front_success & rear_success
        if both_success.any():
            front_pose = front_body_pose_9d_t[:, step_index]
            rear_pose = rear_body_pose_9d_t[:, step_index]
            front_pos_error = torch.linalg.norm(front_pose[:, :3] - prev_pose[:, :3], dim=-1)
            rear_pos_error = torch.linalg.norm(rear_pose[:, :3] - prev_pose[:, :3], dim=-1)
            front_ori_error = _rotation_angle_between_6d(front_pose[:, 3:], prev_pose[:, 3:])
            rear_ori_error = _rotation_angle_between_6d(rear_pose[:, 3:], prev_pose[:, 3:])
            front_inertia = -(front_pos_error + front_ori_error)
            rear_inertia = -(rear_pos_error + rear_ori_error)
            front_bonus = torch.where(prev_view == 0, torch.full_like(front_inertia, 0.5), torch.zeros_like(front_inertia))
            rear_bonus = torch.where(prev_view == 1, torch.full_like(rear_inertia, 0.5), torch.zeros_like(rear_inertia))
            front_total = front_score[:, step_index] + torch.where(prev_valid, front_inertia + front_bonus, torch.zeros_like(front_inertia))
            rear_total = rear_score[:, step_index] + torch.where(prev_valid, rear_inertia + rear_bonus, torch.zeros_like(rear_inertia))
            choose_front = both_success & (front_total >= rear_total)
            choose_rear = both_success & ~choose_front
            current = torch.where(choose_front, torch.zeros_like(current), current)
            current = torch.where(choose_rear, torch.ones_like(current), current)

        neither_success = ~front_success & ~rear_success
        if neither_success.any():
            choose_front = neither_success & front_valid[:, step_index] & (
                ~rear_valid[:, step_index] | (front_score[:, step_index] >= rear_score[:, step_index])
            )
            choose_rear = neither_success & rear_valid[:, step_index] & (~front_valid[:, step_index] | ~choose_front)
            current = torch.where(choose_front, torch.zeros_like(current), current)
            current = torch.where(choose_rear, torch.ones_like(current), current)

        selected[:, step_index] = current
        changed[:, step_index] = (current < 2) & (prev_view < 2) & (current != prev_view)

    return TemporalVisualRoutingOutput(selected_view_index_t=selected, selected_view_changed_t=changed)


def reroute_vision_output_with_inertia(
    vision_output: SingleStepVisionOutput,
    *,
    batch_size: int,
    steps: int,
    device: torch.device,
) -> SingleStepVisionOutput:
    routing = route_selected_view_with_inertia_t(
        front_pnp_success_t=vision_output.front_candidate.pnp_success.reshape(batch_size, steps),
        rear_pnp_success_t=vision_output.rear_candidate.pnp_success.reshape(batch_size, steps),
        front_v_sup_t=vision_output.front_candidate.v_sup.reshape(batch_size, steps),
        rear_v_sup_t=vision_output.rear_candidate.v_sup.reshape(batch_size, steps),
        front_v_rep_t=vision_output.front_candidate.v_rep.reshape(batch_size, steps),
        rear_v_rep_t=vision_output.rear_candidate.v_rep.reshape(batch_size, steps),
        front_pred_target_area_t=vision_output.front_candidate.pred_target_area.reshape(batch_size, steps),
        rear_pred_target_area_t=vision_output.rear_candidate.pred_target_area.reshape(batch_size, steps),
        front_body_pose_9d_t=vision_output.front_candidate.body_pose_9d.reshape(batch_size, steps, 9),
        rear_body_pose_9d_t=vision_output.rear_candidate.body_pose_9d.reshape(batch_size, steps, 9),
    )
    routed_selected_view_index = routing.selected_view_index_t.reshape(batch_size * steps)
    routed_selected_view_onehot = F.one_hot(
        routed_selected_view_index,
        num_classes=3,
    ).to(dtype=vision_output.selected_view_onehot.dtype, device=device)
    routed_selected_candidate = SelectedVisualCandidateOutput(
        view_index=routed_selected_view_index,
        view_onehot=routed_selected_view_onehot,
        view_changed=routing.selected_view_changed_t.reshape(batch_size * steps),
        visual_embedding=_select_view_tensor(
            vision_output.front_candidate.visual_embedding,
            vision_output.rear_candidate.visual_embedding,
            routed_selected_view_index,
        ),
        raw_visual_evidence_strength=_select_view_tensor(
            vision_output.front_candidate.raw_visual_evidence_strength,
            vision_output.rear_candidate.raw_visual_evidence_strength,
            routed_selected_view_index,
        ),
        keypoints_xy=_select_view_tensor(
            vision_output.front_candidate.keypoints_xy,
            vision_output.rear_candidate.keypoints_xy,
            routed_selected_view_index,
        ),
        keypoint_support=_select_view_tensor(
            vision_output.front_candidate.keypoint_support,
            vision_output.rear_candidate.keypoint_support,
            routed_selected_view_index,
        ),
        pnp_success=_select_view_tensor(
            vision_output.front_candidate.pnp_success,
            vision_output.rear_candidate.pnp_success,
            routed_selected_view_index,
        ),
        reprojection_error=_select_view_tensor(
            vision_output.front_candidate.reprojection_error,
            vision_output.rear_candidate.reprojection_error,
            routed_selected_view_index,
        ),
        v_sup=_select_view_tensor(
            vision_output.front_candidate.v_sup,
            vision_output.rear_candidate.v_sup,
            routed_selected_view_index,
        ),
        v_rep=_select_view_tensor(
            vision_output.front_candidate.v_rep,
            vision_output.rear_candidate.v_rep,
            routed_selected_view_index,
        ),
        pred_target_area=_select_view_tensor(
            vision_output.front_candidate.pred_target_area,
            vision_output.rear_candidate.pred_target_area,
            routed_selected_view_index,
        ),
        view_valid=(routed_selected_view_index < 2).to(
            dtype=vision_output.front_candidate.view_valid.dtype
        ),
        pos_valid=_select_view_tensor(
            vision_output.front_candidate.pos_valid,
            vision_output.rear_candidate.pos_valid,
            routed_selected_view_index,
        ),
        ori_valid=_select_view_tensor(
            vision_output.front_candidate.ori_valid,
            vision_output.rear_candidate.ori_valid,
            routed_selected_view_index,
        ),
        body_pose_9d=_select_view_tensor(
            vision_output.front_candidate.body_pose_9d,
            vision_output.rear_candidate.body_pose_9d,
            routed_selected_view_index,
        ),
    )
    return SingleStepVisionOutput(
        front_candidate=vision_output.front_candidate,
        rear_candidate=vision_output.rear_candidate,
        selected_candidate=routed_selected_candidate,
        front_segmentation_logits=vision_output.front_segmentation_logits,
        rear_segmentation_logits=vision_output.rear_segmentation_logits,
        front_keypoints_xy=vision_output.front_keypoints_xy,
        rear_keypoints_xy=vision_output.rear_keypoints_xy,
        front_pnp_success=vision_output.front_pnp_success,
        rear_pnp_success=vision_output.rear_pnp_success,
        front_reprojection_error=vision_output.front_reprojection_error,
        rear_reprojection_error=vision_output.rear_reprojection_error,
        front_v_sup=vision_output.front_v_sup,
        rear_v_sup=vision_output.rear_v_sup,
        front_v_rep=vision_output.front_v_rep,
        rear_v_rep=vision_output.rear_v_rep,
        front_raw_visual_evidence_strength=vision_output.front_raw_visual_evidence_strength,
        rear_raw_visual_evidence_strength=vision_output.rear_raw_visual_evidence_strength,
        raw_visual_evidence_strength=routed_selected_candidate.raw_visual_evidence_strength,
        visual_embedding=routed_selected_candidate.visual_embedding,
        front_voting_field=vision_output.front_voting_field,
        rear_voting_field=vision_output.rear_voting_field,
        front_keypoint_support=vision_output.front_keypoint_support,
        rear_keypoint_support=vision_output.rear_keypoint_support,
        selected_view_index=routed_selected_view_index,
        selected_view_onehot=routed_selected_view_onehot,
        selected_view_changed=routed_selected_candidate.view_changed,
        front_pred_target_area=vision_output.front_pred_target_area,
        rear_pred_target_area=vision_output.rear_pred_target_area,
    )


def _select_view_tensor(front_value: Tensor, rear_value: Tensor, selected_view_index: Tensor) -> Tensor:
    if front_value.shape != rear_value.shape:
        raise ValueError("front_value and rear_value must share shape")
    if selected_view_index.ndim != 1 or selected_view_index.shape[0] != front_value.shape[0]:
        raise ValueError("selected_view_index must be [B] and match batch dim")
    selected = torch.zeros_like(front_value)
    front_mask = selected_view_index == 0
    rear_mask = selected_view_index == 1
    if front_mask.any():
        selected[front_mask] = front_value[front_mask]
    if rear_mask.any():
        selected[rear_mask] = rear_value[rear_mask]
    return selected


def _make_pose(position_x: float) -> Tensor:
    return torch.tensor([position_x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)


def test_route_selected_view_with_inertia_prefers_successful_side() -> None:
    output = route_selected_view_with_inertia_t(
        front_pnp_success_t=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        rear_pnp_success_t=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        front_v_sup_t=torch.tensor([[0.9, 0.9]], dtype=torch.float32),
        rear_v_sup_t=torch.tensor([[0.9, 0.9]], dtype=torch.float32),
        front_v_rep_t=torch.tensor([[0.9, 0.9]], dtype=torch.float32),
        rear_v_rep_t=torch.tensor([[0.9, 0.9]], dtype=torch.float32),
        front_pred_target_area_t=torch.tensor([[10.0, 10.0]], dtype=torch.float32),
        rear_pred_target_area_t=torch.tensor([[1000.0, 1000.0]], dtype=torch.float32),
        front_body_pose_9d_t=torch.stack(
            [_make_pose(1.0), _make_pose(1.1)],
            dim=0,
        ).unsqueeze(0),
        rear_body_pose_9d_t=torch.stack(
            [_make_pose(10.0), _make_pose(10.1)],
            dim=0,
        ).unsqueeze(0),
    )
    assert output.selected_view_index_t.tolist() == [[0, 0]]
    assert output.selected_view_changed_t.tolist() == [[False, False]]


def test_route_selected_view_with_inertia_prefers_continuous_pose_when_both_succeed() -> None:
    output = route_selected_view_with_inertia_t(
        front_pnp_success_t=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        rear_pnp_success_t=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        front_v_sup_t=torch.tensor([[0.9, 0.9]], dtype=torch.float32),
        rear_v_sup_t=torch.tensor([[0.9, 0.9]], dtype=torch.float32),
        front_v_rep_t=torch.tensor([[0.9, 0.9]], dtype=torch.float32),
        rear_v_rep_t=torch.tensor([[0.9, 0.9]], dtype=torch.float32),
        front_pred_target_area_t=torch.tensor([[100.0, 100.0]], dtype=torch.float32),
        rear_pred_target_area_t=torch.tensor([[100.0, 100.0]], dtype=torch.float32),
        front_body_pose_9d_t=torch.stack(
            [_make_pose(1.0), _make_pose(1.1)],
            dim=0,
        ).unsqueeze(0),
        rear_body_pose_9d_t=torch.stack(
            [_make_pose(10.0), _make_pose(10.3)],
            dim=0,
        ).unsqueeze(0),
    )
    assert output.selected_view_index_t.tolist() == [[0, 0]]
    assert output.selected_view_changed_t.tolist() == [[False, False]]
