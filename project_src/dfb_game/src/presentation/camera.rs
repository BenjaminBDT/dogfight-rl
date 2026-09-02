use bevy::{
    camera::ClearColorConfig,
    camera::Viewport,
    core_pipeline::core_3d::graph::Core3d,
    pbr::{DistanceFog, FogFalloff},
    prelude::*,
    render::camera::CameraRenderGraph,
    window::{MonitorSelection, PrimaryWindow, WindowMode},
};

use crate::core::config::RepositoryConfig;
use crate::input::actions::ControlBindings;
use crate::presentation::hud::ObservedAircraftRole;
use crate::simulation::components::{AircraftRole, AircraftState};

#[derive(Component)]
pub struct FollowPlayerCamera {
    pub offset: Vec3,
    pub rear_view_offset: Vec3,
}

#[derive(Component)]
pub struct MainViewCamera;

fn config_color(color: [f32; 4]) -> Color {
    Color::srgba(color[0], color[1], color[2], color[3])
}

fn configured_aspect_ratio(camera_config: &crate::core::config::CameraConfig) -> f32 {
    let width = camera_config.aspect_width.max(1);
    let height = camera_config.aspect_height.max(1);
    width as f32 / height as f32
}

pub fn resolve_follow_camera_pose(
    player_transform: &Transform,
    follow: &FollowPlayerCamera,
    rear_view: bool,
) -> (Vec3, Vec3, Vec3) {
    let offset = if rear_view {
        follow.rear_view_offset
    } else {
        follow.offset
    };
    let forward = player_transform.rotation * Vec3::Z;
    let up = player_transform.rotation * Vec3::Y;
    let desired_position = player_transform.translation + player_transform.rotation * offset;
    let look_direction = if rear_view { -forward } else { forward };
    (desired_position, look_direction, up)
}

pub fn spawn_camera(mut commands: Commands, config: Res<RepositoryConfig>) {
    let camera_config = &config.game.camera;
    commands.spawn((
        Camera {
            clear_color: ClearColorConfig::Custom(config_color(camera_config.sky_color)),
            ..default()
        },
        CameraRenderGraph::new(Core3d),
        Camera3d::default(),
        Projection::from(PerspectiveProjection {
            fov: camera_config.fov_y_degrees.to_radians(),
            aspect_ratio: configured_aspect_ratio(camera_config),
            ..default()
        }),
        DistanceFog {
            color: config_color(camera_config.fog_color),
            directional_light_color: config_color(camera_config.fog_light_color),
            directional_light_exponent: camera_config.fog_light_exponent,
            falloff: FogFalloff::from_visibility_color(
                camera_config.fog_visibility,
                config_color(camera_config.fog_color),
            ),
        },
        Transform::from_xyz(-20.0, 6.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
        FollowPlayerCamera {
            offset: Vec3::from_array(camera_config.follow_offset),
            rear_view_offset: Vec3::from_array(camera_config.rear_view_offset),
        },
        MainViewCamera,
    ));
}

pub fn sync_camera_projection(
    config: Res<RepositoryConfig>,
    window_query: Option<Single<&Window, With<PrimaryWindow>>>,
    mut camera_query: Query<(&mut Camera, &mut Projection), With<MainViewCamera>>,
) {
    let camera_config = &config.game.camera;
    let target_aspect = configured_aspect_ratio(camera_config);
    let window_size = window_query.as_ref().map(|window| {
        UVec2::new(
            window.resolution.physical_width(),
            window.resolution.physical_height(),
        )
    });
    for (mut camera, mut projection) in &mut camera_query {
        if let Projection::Perspective(perspective) = projection.as_mut() {
            perspective.fov = camera_config.fov_y_degrees.to_radians();
            perspective.aspect_ratio = target_aspect;
        }
        if let (Some(window), Some(window_size)) = (window_query.as_ref(), window_size) {
            let _ = window;
            camera.viewport = Some(fixed_aspect_viewport(window_size, target_aspect));
        }
    }
}

pub fn toggle_fullscreen(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    window_query: Option<Single<&mut Window, With<PrimaryWindow>>>,
) {
    let (Some(keyboard), Some(mut window)) = (keyboard, window_query) else {
        return;
    };
    if !keyboard.just_pressed(KeyCode::F11) {
        return;
    }
    window.mode = match window.mode {
        WindowMode::BorderlessFullscreen(_) | WindowMode::Fullscreen(_, _) => WindowMode::Windowed,
        _ => WindowMode::BorderlessFullscreen(MonitorSelection::Current),
    };
}

pub fn update_follow_camera(
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mouse_buttons: Option<Res<ButtonInput<MouseButton>>>,
    bindings: Option<Res<ControlBindings>>,
    observed_role: Option<Res<ObservedAircraftRole>>,
    player_query: Query<(&AircraftRole, &AircraftState, &Transform), Without<MainViewCamera>>,
    mut camera_query: Query<(&FollowPlayerCamera, &mut Transform), With<MainViewCamera>>,
) {
    let (Some(keyboard), Some(mouse_buttons), Some(bindings)) = (keyboard, mouse_buttons, bindings)
    else {
        return;
    };
    let observed_role = observed_role
        .as_ref()
        .map(|role| role.0)
        .unwrap_or(AircraftRole::Fighter1);
    let Some((_, _player, player_transform)) = player_query
        .iter()
        .find(|(role, state, _)| **role == observed_role && !state.is_destroyed)
    else {
        return;
    };

    for (follow, mut transform) in &mut camera_query {
        let rear_view = bindings.rear_view.pressed(&keyboard, &mouse_buttons);
        let (desired_position, look_direction, up) =
            resolve_follow_camera_pose(player_transform, follow, rear_view);
        transform.translation = desired_position;
        transform.look_to(look_direction, up);
    }
}

fn fixed_aspect_viewport(window_size: UVec2, target_aspect: f32) -> Viewport {
    let window_width = window_size.x.max(1);
    let window_height = window_size.y.max(1);
    let window_aspect = window_width as f32 / window_height as f32;
    let (viewport_size, viewport_position) = if window_aspect > target_aspect {
        let height = window_height;
        let width = ((height as f32) * target_aspect)
            .round()
            .clamp(1.0, window_width as f32) as u32;
        let x = (window_width.saturating_sub(width)) / 2;
        (UVec2::new(width, height), UVec2::new(x, 0))
    } else {
        let width = window_width;
        let height = ((width as f32) / target_aspect)
            .round()
            .clamp(1.0, window_height as f32) as u32;
        let y = (window_height.saturating_sub(height)) / 2;
        (UVec2::new(width, height), UVec2::new(0, y))
    };
    Viewport {
        physical_position: viewport_position,
        physical_size: viewport_size,
        ..default()
    }
}
