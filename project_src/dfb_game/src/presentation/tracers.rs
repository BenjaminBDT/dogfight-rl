use bevy::prelude::*;

use crate::gameplay::combat::CombatPresentationQueue;
use crate::simulation::components::AircraftRole;

#[derive(Component)]
pub struct TracerLifetime {
    pub remaining_seconds: f32,
}

pub fn drain_presentation_queue(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut presentation_queue: ResMut<CombatPresentationQueue>,
) {
    let destroyed_events = std::mem::take(&mut presentation_queue.destroyed);
    for event in destroyed_events {
        let color = match event.role {
            AircraftRole::Fighter1 => Color::srgb(0.96, 0.62, 0.16),
            AircraftRole::Fighter2 => Color::srgb(0.88, 0.24, 0.24),
        };

        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(11.5).mesh().uv(28, 18))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color.with_alpha(0.78),
                emissive: color.into(),
                unlit: true,
                ..default()
            })),
            Transform::from_translation(event.position),
            TracerLifetime {
                remaining_seconds: 0.28,
            },
        ));
    }
}

pub fn update_tracer_lifetimes(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut TracerLifetime)>,
) {
    let delta = time.delta_secs();
    for (entity, mut lifetime) in &mut query {
        lifetime.remaining_seconds -= delta;
        if lifetime.remaining_seconds <= 0.0 {
            commands.entity(entity).despawn();
        }
    }
}
