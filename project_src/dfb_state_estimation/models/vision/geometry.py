from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math

import cv2
import numpy as np
import torch
from torch import Tensor


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_keypoint_schema_path() -> str:
    return str(
        _repo_root()
        / "config/dfb_state_estimation/keypoints/fighter_surface_fps8_plus_center_v1.json"
    )


@dataclass(frozen=True)
class GeometryValidationConfig:
    enabled: bool = True
    fov_y_degrees: float = 105.0
    reprojection_error_half: float = 4.0
    pnp_success_reprojection_threshold: float = 8.0
    min_pnp_points: int = 4
    pnp_top_k_points: int = 6
    pnp_success_min_selected_support_mean: float = 0.6
    pnp_success_min_camera_depth: float = 1.0
    pnp_success_max_camera_depth: float = 10000.0
    pnp_success_max_camera_translation_norm: float = 10000.0
    keypoint_schema_path: str = field(default_factory=_default_keypoint_schema_path)


@dataclass(frozen=True)
class CameraGeometryOutput:
    pnp_success: Tensor
    reprojection_error: Tensor
    v_sup: Tensor
    v_rep: Tensor
    raw_visual_evidence_strength: Tensor
    camera_pose_6d: Tensor


@dataclass(frozen=True)
class CameraProjectionOutput:
    pnp_success: Tensor
    reprojection_error: Tensor
    projected_keypoints_px: Tensor


class VisualGeometryValidator:
    def __init__(self, config: GeometryValidationConfig) -> None:
        self.config = config
        self._object_points = self._load_object_points(config.keypoint_schema_path)

    @property
    def num_keypoints(self) -> int:
        return int(self._object_points.shape[0])

    def evaluate_batch(
        self,
        keypoints_xy: Tensor,
        keypoint_support: Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> CameraGeometryOutput:
        if keypoints_xy.ndim != 3 or keypoints_xy.shape[-1] != 2:
            raise ValueError("keypoints_xy must be [B, K, 2]")
        if keypoint_support.shape != keypoints_xy.shape[:2]:
            raise ValueError("keypoint_support must match keypoints_xy batch/keypoint dims")
        if keypoints_xy.shape[1] != self.num_keypoints:
            raise ValueError(
                f"expected {self.num_keypoints} keypoints, got {keypoints_xy.shape[1]}"
            )

        support = keypoint_support.clamp(0.0, 1.0)
        selected_support = self._selected_support_batch(support)
        v_sup = selected_support.mean(dim=1)
        if not self.config.enabled:
            zeros = torch.zeros_like(v_sup)
            infs = torch.full_like(v_sup, float("inf"))
            return CameraGeometryOutput(
                pnp_success=zeros,
                reprojection_error=infs,
                v_sup=v_sup,
                v_rep=zeros,
                raw_visual_evidence_strength=0.5 * v_sup,
                camera_pose_6d=torch.zeros(
                    keypoints_xy.shape[0],
                    6,
                    dtype=keypoints_xy.dtype,
                    device=keypoints_xy.device,
                ),
            )

        keypoints_px = self._normalized_to_pixel_coordinates(
            keypoints_xy.detach(), image_height=image_height, image_width=image_width
        )
        intrinsics = self._camera_matrix(
            image_height=image_height,
            image_width=image_width,
            fov_y_degrees=self.config.fov_y_degrees,
        )
        reprojection_errors: list[float] = []
        pnp_successes: list[float] = []
        camera_poses: list[np.ndarray] = []
        for batch_index in range(keypoints_px.shape[0]):
            support_np = support[batch_index].detach().cpu().numpy()
            success, reproj_error, _, camera_pose_6d = self._solve_pnp_projection(
                keypoints_px[batch_index].detach().cpu().numpy(),
                support_np,
                intrinsics,
            )
            strict_success = success and self._is_strict_pnp_success(
                support=support_np,
                reprojection_error=reproj_error,
                camera_pose_6d=camera_pose_6d,
            )
            pnp_successes.append(1.0 if strict_success else 0.0)
            reprojection_errors.append(reproj_error)
            if strict_success and np.isfinite(camera_pose_6d).all():
                camera_poses.append(camera_pose_6d)
            else:
                camera_poses.append(np.zeros((6,), dtype=np.float32))

        reprojection_error_tensor = keypoints_xy.new_tensor(reprojection_errors)
        pnp_success_tensor = keypoints_xy.new_tensor(pnp_successes)
        finite_error = torch.nan_to_num(
            reprojection_error_tensor,
            nan=float("inf"),
            posinf=float("inf"),
            neginf=float("inf"),
        )
        v_rep = torch.exp(
            -math.log(2.0)
            * (finite_error / self.config.reprojection_error_half).pow(2)
        ) * pnp_success_tensor
        raw_visual_evidence_strength = 0.5 * v_sup + 0.5 * v_rep
        return CameraGeometryOutput(
            pnp_success=pnp_success_tensor,
            reprojection_error=reprojection_error_tensor,
            v_sup=v_sup,
            v_rep=v_rep,
            raw_visual_evidence_strength=raw_visual_evidence_strength,
            camera_pose_6d=keypoints_xy.new_tensor(np.stack(camera_poses, axis=0)),
        )

    def estimate_projection_batch(
        self,
        keypoints_xy: Tensor,
        keypoint_support: Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> CameraProjectionOutput:
        if keypoints_xy.ndim != 3 or keypoints_xy.shape[-1] != 2:
            raise ValueError("keypoints_xy must be [B, K, 2]")
        if keypoint_support.shape != keypoints_xy.shape[:2]:
            raise ValueError("keypoint_support must match keypoints_xy batch/keypoint dims")
        keypoints_px = self._normalized_to_pixel_coordinates(
            keypoints_xy.detach(), image_height=image_height, image_width=image_width
        )
        support = keypoint_support.clamp(0.0, 1.0)
        intrinsics = self._camera_matrix(
            image_height=image_height,
            image_width=image_width,
            fov_y_degrees=self.config.fov_y_degrees,
        )
        reprojection_errors: list[float] = []
        pnp_successes: list[float] = []
        projected_points: list[np.ndarray] = []
        for batch_index in range(keypoints_px.shape[0]):
            support_np = support[batch_index].detach().cpu().numpy()
            success, reproj_error, projected, camera_pose_6d = self._solve_pnp_projection(
                keypoints_px[batch_index].detach().cpu().numpy(),
                support_np,
                intrinsics,
            )
            strict_success = success and self._is_strict_pnp_success(
                support=support_np,
                reprojection_error=reproj_error,
                camera_pose_6d=camera_pose_6d,
            )
            pnp_successes.append(1.0 if strict_success else 0.0)
            reprojection_errors.append(reproj_error)
            projected_points.append(projected)
        return CameraProjectionOutput(
            pnp_success=keypoints_xy.new_tensor(pnp_successes),
            reprojection_error=keypoints_xy.new_tensor(reprojection_errors),
            projected_keypoints_px=keypoints_xy.new_tensor(np.stack(projected_points, axis=0)),
        )

    def _solve_pnp_projection(
        self,
        image_points_px: np.ndarray,
        support: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[bool, float, np.ndarray, np.ndarray]:
        image_points_px = image_points_px.astype(np.float32, copy=False)
        support = support.astype(np.float32, copy=False)
        selected = self._top_support_indices(support)
        if selected is None:
            return False, float("inf"), np.zeros_like(image_points_px), np.zeros((6,), dtype=np.float32)
        object_points = self._object_points[selected].astype(np.float32, copy=False)
        image_points = image_points_px[selected].astype(np.float32, copy=False)
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                intrinsics,
                None,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success:
                return False, float("inf"), np.zeros_like(image_points_px), np.zeros((6,), dtype=np.float32)
            # Refine with iterative PnP for a stabler reprojection estimate.
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                intrinsics,
                None,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return False, float("inf"), np.zeros_like(image_points_px), np.zeros((6,), dtype=np.float32)
            projected, _ = cv2.projectPoints(
                self._object_points,
                rvec,
                tvec,
                intrinsics,
                None,
            )
        except cv2.error:
            return False, float("inf"), np.zeros_like(image_points_px), np.zeros((6,), dtype=np.float32)

        projected = projected.reshape(-1, 2)
        point_errors = np.linalg.norm(projected - image_points_px, axis=1)
        weights = np.clip(support, 1e-6, None)
        reprojection_error = float(np.dot(point_errors, weights) / weights.sum())
        camera_pose_6d = np.concatenate(
            [
                rvec.reshape(3).astype(np.float64, copy=False),
                tvec.reshape(3).astype(np.float64, copy=False),
            ],
            axis=0,
        )
        if (not np.isfinite(camera_pose_6d).all()) or np.abs(camera_pose_6d).max(initial=0.0) > 1.0e6:
            camera_pose_6d = np.zeros((6,), dtype=np.float32)
        else:
            camera_pose_6d = camera_pose_6d.astype(np.float32, copy=False)
        return True, reprojection_error, projected, camera_pose_6d

    def _top_support_indices(self, support: np.ndarray) -> np.ndarray | None:
        if support.size < self.config.min_pnp_points:
            return None
        ordering = np.argsort(-support, kind="stable")
        top_k = min(self.config.pnp_top_k_points, support.size)
        selected = ordering[:top_k]
        if selected.size < self.config.min_pnp_points:
            return None
        return selected

    def _is_strict_pnp_success(
        self,
        *,
        support: np.ndarray,
        reprojection_error: float,
        camera_pose_6d: np.ndarray,
    ) -> bool:
        selected = self._top_support_indices(support)
        if selected is None:
            return False
        selected_support = support[selected]
        support_mean = float(np.clip(selected_support, 0.0, 1.0).mean()) if selected_support.size > 0 else 0.0
        if support_mean < self.config.pnp_success_min_selected_support_mean:
            return False
        if reprojection_error > self.config.pnp_success_reprojection_threshold:
            return False
        if not np.isfinite(camera_pose_6d).all():
            return False
        translation = camera_pose_6d[3:].astype(np.float64, copy=False)
        translation_norm = float(np.linalg.norm(translation))
        if translation_norm > self.config.pnp_success_max_camera_translation_norm:
            return False
        depth = float(translation[2])
        if depth < self.config.pnp_success_min_camera_depth:
            return False
        if depth > self.config.pnp_success_max_camera_depth:
            return False
        return True

    def _selected_support_batch(self, support: Tensor) -> Tensor:
        top_k = min(self.config.pnp_top_k_points, support.shape[1])
        selected, _ = torch.topk(support, k=top_k, dim=1)
        return selected

    @staticmethod
    def _normalized_to_pixel_coordinates(
        keypoints_xy: Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> Tensor:
        scale = keypoints_xy.new_tensor(
            [max(image_width - 1, 1), max(image_height - 1, 1)]
        )
        return keypoints_xy * scale

    @staticmethod
    def _camera_matrix(
        *,
        image_height: int,
        image_width: int,
        fov_y_degrees: float,
    ) -> np.ndarray:
        tan_half_y = math.tan(math.radians(fov_y_degrees) * 0.5)
        aspect = max(image_width, 1) / max(image_height, 1)
        tan_half_x = tan_half_y * aspect
        fx = max(image_width - 1, 1) / (2.0 * max(tan_half_x, 1e-6))
        fy = max(image_height - 1, 1) / (2.0 * max(tan_half_y, 1e-6))
        cx = max(image_width - 1, 1) * 0.5
        cy = max(image_height - 1, 1) * 0.5
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _load_object_points(schema_path: str) -> np.ndarray:
        path = Path(schema_path)
        if not path.is_absolute():
            path = (_repo_root() / path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        labels = data["point_labels"]
        points = data["points_3d_object"]
        return np.array([points[label] for label in labels], dtype=np.float32)


def _test_config() -> GeometryValidationConfig:
    return GeometryValidationConfig(
        enabled=True,
        pnp_success_reprojection_threshold=8.0,
        min_pnp_points=4,
        pnp_top_k_points=6,
    )


def _test_validator() -> VisualGeometryValidator:
    return VisualGeometryValidator(_test_config())


def test_top_support_indices_requires_min_points() -> None:
    validator = _test_validator()
    assert validator._top_support_indices(np.array([0.9, 0.8, 0.7], dtype=np.float32)) is None
    selected = validator._top_support_indices(
        np.array([0.9, 0.8, 0.7, 0.6, 0.1], dtype=np.float32)
    )
    assert selected is not None
    assert selected.tolist() == [0, 1, 2, 3]


def test_top_support_indices_caps_to_top_k_points() -> None:
    validator = VisualGeometryValidator(
        GeometryValidationConfig(
            enabled=True,
            min_pnp_points=4,
            pnp_top_k_points=4,
        )
    )
    selected = validator._top_support_indices(
        np.array([0.7, 0.95, 0.8, 0.6, 0.9, 0.85], dtype=np.float32)
    )
    assert selected is not None
    assert set(selected.tolist()) == {1, 2, 4, 5}


def test_strict_pnp_success_rejects_large_reprojection_error() -> None:
    validator = _test_validator()
    support = np.array([0.9, 0.8, 0.7, 0.6, 0.1], dtype=np.float32)
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0], dtype=np.float32)
    assert validator._is_strict_pnp_success(
        support=support,
        reprojection_error=4.0,
        camera_pose_6d=pose,
    )
    assert not validator._is_strict_pnp_success(
        support=support,
        reprojection_error=12.0,
        camera_pose_6d=pose,
    )


def test_strict_pnp_success_rejects_implausible_camera_pose() -> None:
    validator = _test_validator()
    support = np.array([0.9, 0.8, 0.7, 0.6, 0.1], dtype=np.float32)
    too_far = np.array([0.0, 0.0, 0.0, 50000.0, 0.0, 50000.0], dtype=np.float32)
    behind_camera = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -10.0], dtype=np.float32)
    assert not validator._is_strict_pnp_success(
        support=support,
        reprojection_error=2.0,
        camera_pose_6d=too_far,
    )
    assert not validator._is_strict_pnp_success(
        support=support,
        reprojection_error=2.0,
        camera_pose_6d=behind_camera,
    )
