use bevy::prelude::*;

use crate::core::config::RepositoryConfig;
use crate::gameplay::damage::{AircraftDamageState, DamageStage, subsystem_hitboxes};
use crate::simulation::components::AircraftState;

pub(crate) const AIRCRAFT_COLLISION_RADIUS: f32 = 5.4;

#[derive(Debug, Clone, Copy)]
pub struct AircraftCollisionBox {
    pub name: &'static str,
    pub center: Vec3,
    pub half_extents: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub struct WorldCollisionBox {
    pub name: &'static str,
    pub center: Vec3,
    pub axes: [Vec3; 3],
    pub half_extents: Vec3,
}

#[derive(Component, Debug, Clone, Copy)]
pub struct ObstacleCollider {
    pub half_extents: Vec3,
    pub damage_scale: f32,
}

impl AircraftCollisionBox {
    pub fn to_world(
        self,
        aircraft_position: Vec3,
        aircraft_orientation: Quat,
    ) -> WorldCollisionBox {
        let axes = [
            (aircraft_orientation * Vec3::X).normalize_or_zero(),
            (aircraft_orientation * Vec3::Y).normalize_or_zero(),
            (aircraft_orientation * Vec3::Z).normalize_or_zero(),
        ];
        WorldCollisionBox {
            name: self.name,
            center: aircraft_position + aircraft_orientation * self.center,
            axes,
            half_extents: self.half_extents,
        }
    }
}

impl WorldCollisionBox {
    pub fn from_aabb(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            name: "Aabb",
            center,
            axes: [Vec3::X, Vec3::Y, Vec3::Z],
            half_extents,
        }
    }

    pub fn support_point(&self, direction: Vec3) -> Vec3 {
        let dir = direction.normalize_or_zero();
        let mut point = self.center;
        for axis_idx in 0..3 {
            let axis = self.axes[axis_idx];
            let sign = if axis.dot(dir) >= 0.0 { 1.0 } else { -1.0 };
            point += axis * self.half_extents[axis_idx] * sign;
        }
        point
    }

    pub fn support_radius(&self, direction: Vec3) -> f32 {
        let dir = direction.normalize_or_zero();
        if dir.length_squared() <= f32::EPSILON {
            return self.half_extents.max_element();
        }
        (0..3)
            .map(|axis_idx| self.half_extents[axis_idx] * self.axes[axis_idx].dot(dir).abs())
            .sum()
    }

    pub fn local_point(&self, world_point: Vec3) -> Vec3 {
        let offset = world_point - self.center;
        Vec3::new(
            offset.dot(self.axes[0]),
            offset.dot(self.axes[1]),
            offset.dot(self.axes[2]),
        )
    }

    pub fn vertical_half_extent(&self) -> f32 {
        self.support_radius(Vec3::Y)
    }
}

pub fn aircraft_collision_boxes(damage: &AircraftDamageState) -> Vec<AircraftCollisionBox> {
    let mut boxes = vec![
        AircraftCollisionBox {
            name: "Body",
            center: Vec3::new(0.0, 0.0, 0.0),
            half_extents: Vec3::new(1.4, 0.6, 9.0),
        },
        AircraftCollisionBox {
            name: "Cockpit",
            center: Vec3::new(0.0, 0.55, 0.8),
            half_extents: Vec3::new(0.85, 0.425, 0.9),
        },
    ];
    boxes.extend(
        subsystem_hitboxes()
            .into_iter()
            .filter(|hitbox| damage.subsystem(hitbox.subsystem).stage() != DamageStage::Destroyed)
            .map(|hitbox| AircraftCollisionBox {
                name: hitbox.name,
                center: hitbox.center,
                half_extents: hitbox.half_extents,
            }),
    );
    boxes
}

pub fn aircraft_world_collision_boxes(
    state: &AircraftState,
    damage: &AircraftDamageState,
) -> Vec<WorldCollisionBox> {
    aircraft_world_collision_boxes_for_pose(state.position, state.orientation, damage)
}

pub fn aircraft_world_collision_boxes_for_pose(
    aircraft_position: Vec3,
    aircraft_orientation: Quat,
    damage: &AircraftDamageState,
) -> Vec<WorldCollisionBox> {
    aircraft_collision_boxes(damage)
        .into_iter()
        .map(|local_box| local_box.to_world(aircraft_position, aircraft_orientation))
        .collect()
}

pub fn apply_world_bounds(state: &mut AircraftState, config: &RepositoryConfig) {
    let radial_distance = Vec2::new(state.position.x, state.position.z).length();
    if radial_distance >= config.scene.arena_radius
        && state.out_of_bounds_seconds >= config.scene.out_of_bounds_grace_seconds
    {
        state.is_destroyed = true;
    }
}

pub fn closest_point_on_aabb(point: Vec3, center: Vec3, half_extents: Vec3) -> Vec3 {
    point.clamp(center - half_extents, center + half_extents)
}

pub fn segment_aabb_hit_fraction(
    segment_start: Vec3,
    segment_end: Vec3,
    aabb_center: Vec3,
    aabb_half_extents: Vec3,
) -> Option<f32> {
    let direction = segment_end - segment_start;
    let min = aabb_center - aabb_half_extents;
    let max = aabb_center + aabb_half_extents;

    let mut t_min: f32 = 0.0;
    let mut t_max: f32 = 1.0;

    for axis in 0..3 {
        let start = segment_start[axis];
        let delta = direction[axis];
        let slab_min = min[axis];
        let slab_max = max[axis];

        if delta.abs() <= f32::EPSILON {
            if start < slab_min || start > slab_max {
                return None;
            }
            continue;
        }

        let inv_delta = 1.0 / delta;
        let mut t1 = (slab_min - start) * inv_delta;
        let mut t2 = (slab_max - start) * inv_delta;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
        }

        t_min = t_min.max(t1);
        t_max = t_max.min(t2);
        if t_min > t_max {
            return None;
        }
    }

    Some(t_min.clamp(0.0, 1.0))
}

pub fn segment_obb_hit_fraction(
    segment_start: Vec3,
    segment_end: Vec3,
    obb: &WorldCollisionBox,
    inflation_radius: f32,
) -> Option<f32> {
    let local_start = obb.local_point(segment_start);
    let local_end = obb.local_point(segment_end);
    segment_aabb_hit_fraction(
        local_start,
        local_end,
        Vec3::ZERO,
        obb.half_extents + Vec3::splat(inflation_radius.max(0.0)),
    )
}

pub fn obb_plane_y_penetration(
    obb: &WorldCollisionBox,
    plane_y: f32,
    plane_normal: Vec3,
) -> Option<(Vec3, f32)> {
    let signed_vertical = if plane_normal.y >= 0.0 {
        obb.center.y - obb.vertical_half_extent()
    } else {
        obb.center.y + obb.vertical_half_extent()
    };
    let penetrating = if plane_normal.y >= 0.0 {
        signed_vertical <= plane_y
    } else {
        signed_vertical >= plane_y
    };
    if !penetrating {
        return None;
    }

    let penetration = if plane_normal.y >= 0.0 {
        plane_y - signed_vertical
    } else {
        signed_vertical - plane_y
    };
    let support = obb.support_point(-plane_normal);
    Some((Vec3::new(support.x, plane_y, support.z), penetration))
}

pub fn obb_aabb_penetration(
    obb: &WorldCollisionBox,
    aabb_center: Vec3,
    aabb_half_extents: Vec3,
) -> Option<(Vec3, Vec3, f32)> {
    obb_obb_penetration(
        obb,
        &WorldCollisionBox::from_aabb(aabb_center, aabb_half_extents),
    )
}

pub fn obb_obb_penetration(
    a: &WorldCollisionBox,
    b: &WorldCollisionBox,
) -> Option<(Vec3, Vec3, f32)> {
    let a_ext = [a.half_extents.x, a.half_extents.y, a.half_extents.z];
    let b_ext = [b.half_extents.x, b.half_extents.y, b.half_extents.z];
    let mut rotation = [[0.0f32; 3]; 3];
    let mut abs_rotation = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rotation[i][j] = a.axes[i].dot(b.axes[j]);
            abs_rotation[i][j] = rotation[i][j].abs() + 1e-5;
        }
    }

    let center_delta = b.center - a.center;
    let translation = [
        center_delta.dot(a.axes[0]),
        center_delta.dot(a.axes[1]),
        center_delta.dot(a.axes[2]),
    ];

    let mut best_penetration = f32::INFINITY;
    let mut best_axis = Vec3::X;
    let mut update_best = |axis: Vec3, penetration: f32, sign_hint: f32| {
        if penetration < best_penetration {
            best_penetration = penetration;
            let signed_axis = if sign_hint >= 0.0 { axis } else { -axis };
            best_axis = signed_axis.normalize_or_zero();
        }
    };

    for i in 0..3 {
        let radius_a = a_ext[i];
        let radius_b = b_ext[0] * abs_rotation[i][0]
            + b_ext[1] * abs_rotation[i][1]
            + b_ext[2] * abs_rotation[i][2];
        let distance = translation[i].abs();
        let penetration = radius_a + radius_b - distance;
        if penetration < 0.0 {
            return None;
        }
        update_best(a.axes[i], penetration, translation[i]);
    }

    for j in 0..3 {
        let radius_a = a_ext[0] * abs_rotation[0][j]
            + a_ext[1] * abs_rotation[1][j]
            + a_ext[2] * abs_rotation[2][j];
        let radius_b = b_ext[j];
        let distance = (translation[0] * rotation[0][j]
            + translation[1] * rotation[1][j]
            + translation[2] * rotation[2][j])
            .abs();
        let penetration = radius_a + radius_b - distance;
        if penetration < 0.0 {
            return None;
        }
        update_best(b.axes[j], penetration, center_delta.dot(b.axes[j]));
    }

    for i in 0..3 {
        for j in 0..3 {
            let axis = a.axes[i].cross(b.axes[j]);
            if axis.length_squared() <= 1e-6 {
                continue;
            }
            let radius_a = a_ext[(i + 1) % 3] * abs_rotation[(i + 2) % 3][j]
                + a_ext[(i + 2) % 3] * abs_rotation[(i + 1) % 3][j];
            let radius_b = b_ext[(j + 1) % 3] * abs_rotation[i][(j + 2) % 3]
                + b_ext[(j + 2) % 3] * abs_rotation[i][(j + 1) % 3];
            let distance = (translation[(i + 2) % 3] * rotation[(i + 1) % 3][j]
                - translation[(i + 1) % 3] * rotation[(i + 2) % 3][j])
                .abs();
            let penetration = radius_a + radius_b - distance;
            if penetration < 0.0 {
                return None;
            }
            update_best(axis, penetration, axis.dot(center_delta));
        }
    }

    let normal = if best_axis.length_squared() <= f32::EPSILON {
        Vec3::X
    } else {
        best_axis.normalize()
    };
    let contact_a = a.support_point(normal);
    let contact_b = b.support_point(-normal);
    let contact = (contact_a + contact_b) * 0.5;
    Some((contact, normal, best_penetration.min(f32::MAX)))
}

#[cfg(test)]
mod tests {
    use super::{
        AircraftCollisionBox, WorldCollisionBox, aircraft_collision_boxes, apply_world_bounds,
        closest_point_on_aabb, obb_obb_penetration, obb_plane_y_penetration,
        segment_obb_hit_fraction,
    };
    use crate::core::config::AircraftConfig;
    use crate::core::config::RepositoryConfig;
    use crate::gameplay::damage::AircraftDamageState;
    use crate::simulation::components::AircraftState;
    use crate::simulation::systems::aircraft_performance_from_config;
    use bevy::prelude::{Quat, Vec3};

    #[test]
    fn closest_point_clamps_to_box() {
        let point = Vec3::new(8.0, -4.0, 1.0);
        let closest =
            closest_point_on_aabb(point, Vec3::new(1.0, 2.0, 3.0), Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(closest, Vec3::new(3.0, -1.0, 1.0));
    }

    #[test]
    fn segment_hits_rotated_obb() {
        let obb = AircraftCollisionBox {
            name: "test",
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        }
        .to_world(Vec3::ZERO, Quat::from_rotation_y(0.4));
        assert!(
            segment_obb_hit_fraction(
                Vec3::new(0.0, 0.0, -10.0),
                Vec3::new(0.0, 0.0, 10.0),
                &obb,
                0.0
            )
            .is_some()
        );
    }

    #[test]
    fn obb_overlap_returns_penetration() {
        let a = WorldCollisionBox::from_aabb(Vec3::ZERO, Vec3::splat(1.5));
        let b = WorldCollisionBox::from_aabb(Vec3::new(2.0, 0.0, 0.0), Vec3::splat(1.5));
        let contact = obb_obb_penetration(&a, &b).expect("expected overlap");
        assert!(contact.2 > 0.0);
        assert!(contact.1.x.abs() > 0.8);
    }

    #[test]
    fn plane_penetration_uses_lowest_support_point() {
        let obb = WorldCollisionBox::from_aabb(Vec3::new(0.0, 0.5, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let penetration = obb_plane_y_penetration(&obb, 0.0, Vec3::Y).expect("expected overlap");
        assert!(penetration.1 > 0.4);
    }

    #[test]
    fn world_bounds_uses_cylindrical_radius() {
        let mut config = RepositoryConfig::default();
        config.scene.arena_radius = 100.0;
        config.scene.out_of_bounds_grace_seconds = 10.0;
        let mut state = AircraftState {
            position: Vec3::new(0.0, 10_000.0, 0.0),
            out_of_bounds_seconds: 10.0,
            ..Default::default()
        };

        apply_world_bounds(&mut state, &config);

        assert!(!state.is_destroyed);
    }

    #[test]
    fn world_bounds_destroys_only_after_grace_time_expires() {
        let mut config = RepositoryConfig::default();
        config.scene.arena_radius = 100.0;
        config.scene.out_of_bounds_grace_seconds = 10.0;
        let mut state = AircraftState {
            position: Vec3::new(120.0, 0.0, 0.0),
            out_of_bounds_seconds: 9.5,
            ..Default::default()
        };

        apply_world_bounds(&mut state, &config);
        assert!(!state.is_destroyed);

        state.out_of_bounds_seconds = 10.0;
        apply_world_bounds(&mut state, &config);
        assert!(state.is_destroyed);
    }

    #[test]
    fn aircraft_collision_boxes_use_mesh_aligned_body_bounds() {
        let performance = aircraft_performance_from_config(&AircraftConfig::default_fighter1());
        let damage = AircraftDamageState::new(100.0, &performance);
        let boxes = aircraft_collision_boxes(&damage);
        assert!(boxes.iter().any(|obb| {
            obb.name == "Body"
                && obb.center == Vec3::ZERO
                && obb.half_extents == Vec3::new(1.4, 0.6, 9.0)
        }));
        assert!(boxes.iter().any(|obb| {
            obb.name == "Cockpit"
                && obb.center == Vec3::new(0.0, 0.55, 0.8)
                && obb.half_extents == Vec3::new(0.85, 0.425, 0.9)
        }));
    }
}
