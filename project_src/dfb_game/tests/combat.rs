use bevy::prelude::Vec3;

use dfb_game::gameplay::combat::{segment_aabb_hit_fraction, segment_intersects_sphere};

#[test]
fn moving_segment_hits_target_sphere() {
    assert!(segment_intersects_sphere(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 20.0),
        Vec3::new(0.0, 0.0, 10.0),
        2.0,
    ));
}

#[test]
fn moving_segment_misses_offset_target_sphere() {
    assert!(!segment_intersects_sphere(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 20.0),
        Vec3::new(8.0, 0.0, 10.0),
        2.0,
    ));
}

#[test]
fn moving_segment_hits_obstacle_box() {
    let hit_t = segment_aabb_hit_fraction(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 20.0),
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(2.0, 2.0, 2.0),
    )
    .expect("expected obstacle hit");

    assert!((0.0..=1.0).contains(&hit_t));
    assert!(hit_t < 0.5);
}

#[test]
fn moving_segment_misses_obstacle_box() {
    assert!(
        segment_aabb_hit_fraction(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 20.0),
            Vec3::new(6.0, 0.0, 10.0),
            Vec3::new(2.0, 2.0, 2.0),
        )
        .is_none()
    );
}
