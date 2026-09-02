use bevy::prelude::Vec3;
use dfb_game::simulation::collision::{
    WorldCollisionBox, closest_point_on_aabb, obb_aabb_penetration,
};

#[test]
fn closest_point_is_clamped_to_obstacle_bounds() {
    let closest = closest_point_on_aabb(
        Vec3::new(9.0, -6.0, 1.0),
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(2.0, 3.0, 4.0),
    );

    assert_eq!(closest, Vec3::new(3.0, -1.0, 1.0));
}

#[test]
fn obb_aabb_penetration_reports_overlap_direction() {
    let aircraft_box = WorldCollisionBox::from_aabb(Vec3::new(4.0, 0.0, 0.0), Vec3::splat(1.6));
    let (_, normal, penetration) =
        obb_aabb_penetration(&aircraft_box, Vec3::ZERO, Vec3::new(3.0, 2.0, 2.0))
            .expect("expected overlap");

    assert!(penetration > 0.0);
    assert!(normal.x.abs() > 0.9);
}
