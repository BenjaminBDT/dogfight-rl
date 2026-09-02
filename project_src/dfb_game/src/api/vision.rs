use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ClearColorConfig, RenderTarget};
use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy::image::Image;
use bevy::pbr::{DistanceFog, FogFalloff};
use bevy::prelude::*;
use bevy::render::camera::CameraRenderGraph;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{NodeRunError, RenderGraph, RenderGraphContext, RenderLabel};
use bevy::render::render_resource::TextureFormat;
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode, PollType,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TextureUsages,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::Msaa;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::window::{PrimaryWindow, Window};

use crate::api::types::{
    ObservationCaptureConfig, VisualCaptureVariant, VisualResolutionMode, VisualSensorKind,
};
use crate::core::config::RepositoryConfig;
use crate::presentation::camera::{FollowPlayerCamera, resolve_follow_camera_pose};
use crate::presentation::hud::HudLayer;
use crate::presentation::hud::ObservedAircraftRole;
use crate::simulation::components::{AircraftRole, AircraftState};

fn config_color(color: [f32; 4]) -> Color {
    Color::srgba(color[0], color[1], color[2], color[3])
}

pub const SEMANTIC_RENDER_LAYER: usize = 1;

#[derive(Debug, Clone, Copy, Resource, Default)]
pub struct SemanticCaptureMode(pub bool);

#[derive(Debug, Clone, Copy, Resource)]
pub struct SemanticRenderConfig {
    pub msaa_enabled: bool,
    pub remove_distance_fog: bool,
}

impl Default for SemanticRenderConfig {
    fn default() -> Self {
        Self {
            msaa_enabled: false,
            remove_distance_fog: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CapturedVisualFrame {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    pub generation: u64,
}

#[derive(Debug, Default, Resource)]
pub struct VisualCaptureFrames {
    pub frames: HashMap<VisualCaptureKey, CapturedVisualFrame>,
}

#[derive(Debug, Default, Resource, Clone, Copy)]
pub struct VisualCaptureGeneration(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyntheticCaptureSessionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyntheticCaptureRequestId(pub u64);

#[derive(Debug, Default, Resource)]
pub struct SyntheticCaptureSessions {
    next_session_id: u64,
    next_request_id: u64,
    requests: HashMap<SyntheticCaptureRequestId, SyntheticCaptureRequestMeta>,
    frames: HashMap<(SyntheticCaptureSessionId, VisualCaptureKey), CapturedVisualFrame>,
}

#[derive(Debug, Clone, Copy)]
struct SyntheticCaptureRequestMeta {
    session_id: SyntheticCaptureSessionId,
    key: VisualCaptureKey,
}

#[derive(Debug, Default, Resource)]
pub struct PendingVisualCaptures {
    pub keys: HashSet<VisualCaptureKey>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Resource, Default)]
pub enum VisualCaptureBackend {
    #[default]
    Screenshot,
    Offscreen,
}

#[derive(Debug, Clone, Default, Resource)]
pub struct OffscreenVisualCaptureBridge(
    pub Arc<Mutex<HashMap<VisualCaptureKey, CapturedVisualFrame>>>,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisualCaptureKey {
    pub kind: VisualSensorKind,
    pub variant: VisualCaptureVariant,
}

#[derive(Component, Clone, Copy)]
pub struct ObservationCaptureCamera {
    pub kind: VisualSensorKind,
    pub variant: VisualCaptureVariant,
}

#[derive(Component, Clone)]
pub struct CaptureTargetImage {
    pub handle: Handle<Image>,
    pub width: u32,
    pub height: u32,
}

#[derive(Component, Clone, Copy)]
pub struct PendingVisualCaptureRequest {
    pub key: VisualCaptureKey,
    pub synthetic_session: Option<SyntheticCaptureSessionId>,
    pub synthetic_request: Option<SyntheticCaptureRequestId>,
}

#[derive(Component, Clone)]
pub struct OffscreenImageCopier {
    pub key: VisualCaptureKey,
    pub src_image: Handle<Image>,
    pub buffer: Buffer,
    pub width: u32,
    pub height: u32,
}

impl OffscreenImageCopier {
    fn new(
        key: VisualCaptureKey,
        src_image: Handle<Image>,
        width: u32,
        height: u32,
        render_device: &RenderDevice,
    ) -> Self {
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(width as usize * 4);
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: (padded_bytes_per_row * height as usize) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            key,
            src_image,
            buffer,
            width,
            height,
        }
    }
}

#[derive(Clone, Default, Resource, Deref, DerefMut)]
struct ExtractedOffscreenImageCopiers(pub Vec<OffscreenImageCopier>);

#[derive(Clone)]
struct PendingOffscreenReadback {
    key: VisualCaptureKey,
    width: u32,
    height: u32,
    buffer: Buffer,
    generation: u64,
    status: Arc<Mutex<Option<bool>>>,
}

#[derive(Default, Resource)]
struct PendingOffscreenReadbacks(pub HashMap<VisualCaptureKey, PendingOffscreenReadback>);

#[derive(Debug, PartialEq, Eq, Clone, Hash, RenderLabel)]
struct OffscreenImageCopyLabel;

#[derive(Default)]
struct OffscreenImageCopyNode;

impl SyntheticCaptureSessions {
    fn begin_session(&mut self) -> SyntheticCaptureSessionId {
        let session_id = SyntheticCaptureSessionId(self.next_session_id);
        self.next_session_id = self.next_session_id.wrapping_add(1);
        session_id
    }

    fn register_request(
        &mut self,
        session_id: SyntheticCaptureSessionId,
        key: VisualCaptureKey,
    ) -> SyntheticCaptureRequestId {
        let request_id = SyntheticCaptureRequestId(self.next_request_id);
        self.next_request_id = self.next_request_id.wrapping_add(1);
        self.requests
            .insert(request_id, SyntheticCaptureRequestMeta { session_id, key });
        request_id
    }

    fn clear_session(&mut self, session_id: SyntheticCaptureSessionId) {
        self.requests
            .retain(|_, meta| meta.session_id != session_id);
        self.frames
            .retain(|(frame_session_id, _), _| *frame_session_id != session_id);
    }

    fn record_request_frame(
        &mut self,
        request_id: SyntheticCaptureRequestId,
        frame: CapturedVisualFrame,
    ) {
        let Some(meta) = self.requests.remove(&request_id) else {
            return;
        };
        self.frames.insert((meta.session_id, meta.key), frame);
    }

    fn take_frame(
        &mut self,
        session_id: SyntheticCaptureSessionId,
        key: VisualCaptureKey,
    ) -> Option<CapturedVisualFrame> {
        self.frames.remove(&(session_id, key))
    }

    fn has_frame(&self, session_id: SyntheticCaptureSessionId, key: VisualCaptureKey) -> bool {
        self.frames.contains_key(&(session_id, key))
    }
}

pub fn begin_synthetic_capture_session(world: &mut World) -> SyntheticCaptureSessionId {
    world
        .get_resource_or_insert_with::<SyntheticCaptureSessions>(Default::default)
        .begin_session()
}

pub fn clear_synthetic_capture_session(world: &mut World, session_id: SyntheticCaptureSessionId) {
    if let Some(mut sessions) = world.get_resource_mut::<SyntheticCaptureSessions>() {
        sessions.clear_session(session_id);
    }
}

pub fn register_synthetic_capture_request(
    world: &mut World,
    session_id: SyntheticCaptureSessionId,
    key: VisualCaptureKey,
) -> SyntheticCaptureRequestId {
    world
        .get_resource_or_insert_with::<SyntheticCaptureSessions>(Default::default)
        .register_request(session_id, key)
}

pub fn take_synthetic_capture_frame(
    world: &mut World,
    session_id: SyntheticCaptureSessionId,
    key: VisualCaptureKey,
) -> Option<CapturedVisualFrame> {
    world
        .get_resource_mut::<SyntheticCaptureSessions>()
        .and_then(|mut sessions| sessions.take_frame(session_id, key))
}

pub fn synthetic_capture_session_ready(
    world: &World,
    session_id: SyntheticCaptureSessionId,
    keys: &[VisualCaptureKey],
) -> bool {
    let Some(sessions) = world.get_resource::<SyntheticCaptureSessions>() else {
        return false;
    };
    keys.iter().all(|key| sessions.has_frame(session_id, *key))
}

pub fn ensure_capture_cameras(
    mut commands: Commands,
    capture_config: Res<ObservationCaptureConfig>,
    config: Res<RepositoryConfig>,
    semantic_render_config: Option<Res<SemanticRenderConfig>>,
    images: Option<ResMut<Assets<Image>>>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    capture_query: Query<(Entity, &ObservationCaptureCamera, &CaptureTargetImage)>,
) {
    if !capture_config.enable_visual {
        return;
    }
    let Some(mut images) = images else {
        return;
    };

    let runtime_window_size = window_query
        .iter()
        .next()
        .map(|window| (window.physical_width(), window.physical_height()));
    let aspect_ratio = configured_aspect_ratio(&config.game.camera);

    let semantic_render_config = semantic_render_config
        .as_deref()
        .copied()
        .unwrap_or_default();

    for sensor in &capture_config.visual_sensors {
        let (width, height) = resolved_sensor_size(sensor, runtime_window_size);
        for variant in sensor.requested_capture_variants() {
            let existing = capture_query
                .iter()
                .find(|(_, camera, _)| camera.kind == sensor.kind && camera.variant == variant)
                .map(|(entity, _, image)| (entity, image.clone()));

            let clear_color = match variant {
                VisualCaptureVariant::Rgb => config_color(config.game.camera.sky_color),
                VisualCaptureVariant::Semantic => Color::BLACK,
            };
            let render_layer = match variant {
                VisualCaptureVariant::Rgb => RenderLayers::layer(0),
                VisualCaptureVariant::Semantic => RenderLayers::layer(SEMANTIC_RENDER_LAYER),
            };

            match existing {
                Some((entity, image)) if image.width == width && image.height == height => {}
                Some((entity, _)) => {
                    let image_handle = create_capture_target(&mut images, width, height, variant);
                    let mut entity_commands = commands.entity(entity);
                    entity_commands.insert((
                        Camera {
                            clear_color: ClearColorConfig::Custom(clear_color),
                            ..default()
                        },
                        RenderTarget::Image(image_handle.clone().into()),
                        Projection::from(PerspectiveProjection {
                            fov: config.game.camera.fov_y_degrees.to_radians(),
                            aspect_ratio,
                            ..default()
                        }),
                        CaptureTargetImage {
                            handle: image_handle,
                            width,
                            height,
                        },
                        render_layer,
                    ));
                    configure_semantic_camera(entity_commands, variant, semantic_render_config);
                }
                None => {
                    let image_handle = create_capture_target(&mut images, width, height, variant);
                    let entity_commands = commands.spawn((
                        Camera {
                            clear_color: ClearColorConfig::Custom(clear_color),
                            ..default()
                        },
                        RenderTarget::Image(image_handle.clone().into()),
                        CameraRenderGraph::new(Core3d),
                        Camera3d::default(),
                        Projection::from(PerspectiveProjection {
                            fov: config.game.camera.fov_y_degrees.to_radians(),
                            aspect_ratio,
                            ..default()
                        }),
                        DistanceFog {
                            color: config_color(config.game.camera.fog_color),
                            directional_light_color: config_color(
                                config.game.camera.fog_light_color,
                            ),
                            directional_light_exponent: config.game.camera.fog_light_exponent,
                            falloff: FogFalloff::from_visibility_color(
                                config.game.camera.fog_visibility,
                                config_color(config.game.camera.fog_color),
                            ),
                        },
                        Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
                        ObservationCaptureCamera {
                            kind: sensor.kind,
                            variant,
                        },
                        CaptureTargetImage {
                            handle: image_handle,
                            width,
                            height,
                        },
                        render_layer,
                    ));
                    configure_semantic_camera(entity_commands, variant, semantic_render_config);
                }
            }
        }
    }
}

fn configure_semantic_camera(
    mut entity_commands: EntityCommands<'_>,
    variant: VisualCaptureVariant,
    semantic_render_config: SemanticRenderConfig,
) {
    match variant {
        VisualCaptureVariant::Rgb => {}
        VisualCaptureVariant::Semantic => {
            if semantic_render_config.msaa_enabled {
                entity_commands.insert(Msaa::Sample4);
            } else {
                // Strict semantic capture avoids MSAA edge resolve, which can smear class ids.
                entity_commands.insert(Msaa::Off);
            }
            if semantic_render_config.remove_distance_fog {
                entity_commands.remove::<DistanceFog>();
            }
        }
    }
}

pub fn sync_capture_cameras(
    observed_role: Option<Res<ObservedAircraftRole>>,
    config: Res<RepositoryConfig>,
    aircraft_query: Query<
        (&AircraftRole, &AircraftState, &Transform),
        Without<ObservationCaptureCamera>,
    >,
    mut capture_query: Query<(&ObservationCaptureCamera, &mut Transform)>,
) {
    let observed_role = observed_role
        .as_deref()
        .map(|role| role.0)
        .unwrap_or(AircraftRole::Fighter1);
    let Some((_, _, player_transform)) = aircraft_query
        .iter()
        .find(|(role, state, _)| **role == observed_role && !state.is_destroyed)
    else {
        return;
    };

    let follow = FollowPlayerCamera {
        offset: Vec3::from_array(config.game.camera.follow_offset),
        rear_view_offset: Vec3::from_array(config.game.camera.rear_view_offset),
    };

    for (camera, mut transform) in &mut capture_query {
        let rear_view = matches!(camera.kind, VisualSensorKind::Rear);
        let (desired_position, look_direction, up) =
            resolve_follow_camera_pose(player_transform, &follow, rear_view);
        transform.translation = desired_position;
        transform.look_to(look_direction, up);
    }
}

pub fn request_visual_captures(
    mut commands: Commands,
    backend: Res<VisualCaptureBackend>,
    capture_config: Res<ObservationCaptureConfig>,
    mut pending: ResMut<PendingVisualCaptures>,
    capture_query: Query<(&ObservationCaptureCamera, &CaptureTargetImage)>,
) {
    if !capture_config.enable_visual || *backend == VisualCaptureBackend::Offscreen {
        return;
    }

    for (camera, image) in &capture_query {
        let key = VisualCaptureKey {
            kind: camera.kind,
            variant: camera.variant,
        };
        if pending.keys.contains(&key) {
            continue;
        }
        pending.keys.insert(key);
        commands.spawn((
            Screenshot::image(image.handle.clone()),
            PendingVisualCaptureRequest {
                key,
                synthetic_session: None,
                synthetic_request: None,
            },
        ));
    }
}

pub fn request_synthetic_capture_variant(
    world: &mut World,
    session_id: SyntheticCaptureSessionId,
    variant: VisualCaptureVariant,
) -> Vec<VisualCaptureKey> {
    let captures = {
        let mut query = world.query::<(&ObservationCaptureCamera, &CaptureTargetImage)>();
        query
            .iter(world)
            .filter(|(camera, _)| camera.variant == variant)
            .map(|(camera, image)| {
                (
                    VisualCaptureKey {
                        kind: camera.kind,
                        variant: camera.variant,
                    },
                    image.handle.clone(),
                )
            })
            .collect::<Vec<_>>()
    };

    for (key, image_handle) in &captures {
        let request_id = register_synthetic_capture_request(world, session_id, *key);
        world.spawn((
            Screenshot::image(image_handle.clone()),
            PendingVisualCaptureRequest {
                key: *key,
                synthetic_session: Some(session_id),
                synthetic_request: Some(request_id),
            },
        ));
    }

    captures.into_iter().map(|(key, _)| key).collect()
}

pub fn ensure_offscreen_image_copiers(
    mut commands: Commands,
    backend: Res<VisualCaptureBackend>,
    capture_config: Res<ObservationCaptureConfig>,
    render_device: Option<Res<RenderDevice>>,
    capture_query: Query<(
        Entity,
        &ObservationCaptureCamera,
        &CaptureTargetImage,
        Option<&OffscreenImageCopier>,
    )>,
) {
    if !capture_config.enable_visual || *backend != VisualCaptureBackend::Offscreen {
        return;
    }
    let Some(render_device) = render_device else {
        return;
    };

    for (entity, camera, image, existing) in &capture_query {
        let needs_insert = existing
            .map(|copier| {
                copier.key.kind != camera.kind
                    || copier.key.variant != camera.variant
                    || copier.width != image.width
                    || copier.height != image.height
                    || copier.src_image != image.handle
            })
            .unwrap_or(true);
        if needs_insert {
            commands.entity(entity).insert(OffscreenImageCopier::new(
                VisualCaptureKey {
                    kind: camera.kind,
                    variant: camera.variant,
                },
                image.handle.clone(),
                image.width,
                image.height,
                &render_device,
            ));
        }
    }
}

pub fn attach_hud_to_capture_camera(
    capture_config: Res<ObservationCaptureConfig>,
    capture_query: Query<(Entity, &ObservationCaptureCamera)>,
    mut hud_query: Query<(Entity, Option<&UiTargetCamera>), With<HudLayer>>,
    mut commands: Commands,
) {
    if !capture_config.enable_visual {
        return;
    }

    let preferred_kind = capture_config
        .visual_sensors
        .iter()
        .find(|sensor| sensor.include_hud)
        .map(|sensor| sensor.kind);
    let Some(preferred_kind) = preferred_kind else {
        return;
    };

    let Some(camera_entity) = capture_query
        .iter()
        .find(|(_, camera)| {
            camera.kind == preferred_kind && camera.variant == VisualCaptureVariant::Rgb
        })
        .map(|(entity, _)| entity)
    else {
        return;
    };

    for (hud_entity, target_camera) in &mut hud_query {
        let already_targeted = target_camera
            .map(|target| target.0 == camera_entity)
            .unwrap_or(false);
        if !already_targeted {
            commands
                .entity(hud_entity)
                .insert(UiTargetCamera(camera_entity));
        }
    }
}

pub fn store_visual_capture_screenshot(
    event: On<ScreenshotCaptured>,
    mut commands: Commands,
    request_query: Query<&PendingVisualCaptureRequest>,
    mut frames: ResMut<VisualCaptureFrames>,
    mut synthetic_sessions: Option<ResMut<SyntheticCaptureSessions>>,
    generation: Res<VisualCaptureGeneration>,
    mut pending: ResMut<PendingVisualCaptures>,
) {
    let entity = event.entity;
    let Ok(request) = request_query.get(entity) else {
        return;
    };
    pending.keys.remove(&request.key);

    let Some(data) = event.image.data.clone() else {
        return;
    };

    let frame = CapturedVisualFrame {
        width: event.image.texture_descriptor.size.width,
        height: event.image.texture_descriptor.size.height,
        bytes: data,
        generation: generation.0,
    };
    frames.frames.insert(request.key, frame.clone());
    if let (Some(_session_id), Some(request_id), Some(ref mut sessions)) = (
        request.synthetic_session,
        request.synthetic_request,
        synthetic_sessions.as_mut(),
    ) {
        sessions.record_request_frame(request_id, frame);
    }
    commands.entity(entity).despawn();
}

pub fn shutdown_visual_capture(world: &mut World) {
    if let Some(mut pending) = world.get_resource_mut::<PendingVisualCaptures>() {
        pending.keys.clear();
    }
    if let Some(mut frames) = world.get_resource_mut::<VisualCaptureFrames>() {
        frames.frames.clear();
    }
    if let Some(bridge) = world.get_resource::<OffscreenVisualCaptureBridge>()
        && let Ok(mut pending_frames) = bridge.0.lock()
    {
        pending_frames.clear();
    }
}

pub fn clear_offscreen_visual_frames(world: &mut World) {
    if let Some(mut frames) = world.get_resource_mut::<VisualCaptureFrames>() {
        frames.frames.clear();
    }
    if let Some(bridge) = world.get_resource::<OffscreenVisualCaptureBridge>()
        && let Ok(mut pending_frames) = bridge.0.lock()
    {
        pending_frames.clear();
    }
}

pub fn clear_offscreen_visual_capture_state(app: &mut App) {
    clear_offscreen_visual_frames(app.world_mut());
    let generation = {
        let mut generation = app
            .world_mut()
            .get_resource_or_insert_with::<VisualCaptureGeneration>(Default::default);
        generation.0 = generation.0.wrapping_add(1);
        generation.0
    };
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        if let Some(mut pending_readbacks) = render_app
            .world_mut()
            .get_resource_mut::<PendingOffscreenReadbacks>()
        {
            pending_readbacks.0.clear();
        }
        let mut render_generation = render_app
            .world_mut()
            .get_resource_or_insert_with::<VisualCaptureGeneration>(Default::default);
        render_generation.0 = generation;
    }
}

pub fn offscreen_visual_frames_ready(world: &World) -> bool {
    let Some(capture_config) = world.get_resource::<ObservationCaptureConfig>() else {
        return true;
    };
    if !capture_config.enable_visual {
        return true;
    }
    let Some(frames) = world.get_resource::<VisualCaptureFrames>() else {
        return false;
    };
    let generation = world
        .get_resource::<VisualCaptureGeneration>()
        .map(|generation| generation.0)
        .unwrap_or_default();
    capture_config.visual_sensors.iter().all(|sensor| {
        sensor.requested_capture_variants().iter().all(|variant| {
            frames
                .frames
                .get(&VisualCaptureKey {
                    kind: sensor.kind,
                    variant: *variant,
                })
                .is_some_and(|frame| {
                    frame.width == sensor.width
                        && frame.height == sensor.height
                        && frame.generation == generation
                })
        })
    })
}

pub fn offscreen_visual_variant_frames_ready(world: &World, variant: VisualCaptureVariant) -> bool {
    let Some(capture_config) = world.get_resource::<ObservationCaptureConfig>() else {
        return true;
    };
    if !capture_config.enable_visual {
        return true;
    }
    let Some(frames) = world.get_resource::<VisualCaptureFrames>() else {
        return false;
    };
    let generation = world
        .get_resource::<VisualCaptureGeneration>()
        .map(|generation| generation.0)
        .unwrap_or_default();
    capture_config.visual_sensors.iter().all(|sensor| {
        if !sensor.requested_capture_variants().contains(&variant) {
            return true;
        }
        frames
            .frames
            .get(&VisualCaptureKey {
                kind: sensor.kind,
                variant,
            })
            .is_some_and(|frame| {
                frame.width == sensor.width
                    && frame.height == sensor.height
                    && frame.generation == generation
            })
    })
}

fn configured_aspect_ratio(camera_config: &crate::core::config::CameraConfig) -> f32 {
    let width = camera_config.aspect_width.max(1);
    let height = camera_config.aspect_height.max(1);
    width as f32 / height as f32
}

fn resolved_sensor_size(
    sensor: &crate::api::types::VisualSensorConfig,
    runtime_window_size: Option<(u32, u32)>,
) -> (u32, u32) {
    match sensor.resolution_mode {
        VisualResolutionMode::Fixed => (sensor.width, sensor.height),
        VisualResolutionMode::RuntimeWindow => {
            runtime_window_size.unwrap_or((sensor.width, sensor.height))
        }
    }
}

fn create_capture_target(
    images: &mut Assets<Image>,
    width: u32,
    height: u32,
    variant: VisualCaptureVariant,
) -> Handle<Image> {
    let format = match variant {
        VisualCaptureVariant::Rgb => TextureFormat::Rgba8UnormSrgb,
        // First-stage semantic root fix: keep a normal RGBA render target, but remove sRGB
        // semantics so the capture path can quantize directly to class-id bytes downstream.
        VisualCaptureVariant::Semantic => TextureFormat::Rgba8Unorm,
    };
    let mut image = Image::new_target_texture(width, height, format, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    images.add(image)
}

pub fn install_offscreen_visual_capture(app: &mut App) {
    let bridge = app
        .world()
        .resource::<OffscreenVisualCaptureBridge>()
        .clone();
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };

    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(OffscreenImageCopyLabel, OffscreenImageCopyNode);
    graph.add_node_edge(
        bevy::render::graph::CameraDriverLabel,
        OffscreenImageCopyLabel,
    );

    render_app
        .insert_resource(bridge)
        .init_resource::<VisualCaptureGeneration>()
        .init_resource::<PendingOffscreenReadbacks>()
        .add_systems(
            ExtractSchedule,
            (
                extract_offscreen_image_copiers,
                extract_visual_capture_generation,
            ),
        )
        .add_systems(
            Render,
            receive_offscreen_image_from_buffer.after(RenderSystems::Render),
        );
}

pub fn drain_offscreen_visual_frames(
    backend: Res<VisualCaptureBackend>,
    bridge: Res<OffscreenVisualCaptureBridge>,
    mut frames: ResMut<VisualCaptureFrames>,
) {
    if *backend != VisualCaptureBackend::Offscreen {
        return;
    }
    let Ok(mut pending_frames) = bridge.0.lock() else {
        return;
    };
    if pending_frames.is_empty() {
        return;
    }
    frames.frames.extend(pending_frames.drain());
}

pub fn drain_offscreen_visual_frames_now(world: &mut World) {
    let use_offscreen = world
        .get_resource::<VisualCaptureBackend>()
        .is_some_and(|backend| *backend == VisualCaptureBackend::Offscreen);
    if !use_offscreen {
        return;
    }

    let pending_frames = {
        let Some(bridge) = world.get_resource::<OffscreenVisualCaptureBridge>() else {
            return;
        };
        let Ok(mut pending_frames) = bridge.0.lock() else {
            return;
        };
        if pending_frames.is_empty() {
            return;
        }
        pending_frames.drain().collect::<Vec<_>>()
    };

    let Some(mut frames) = world.get_resource_mut::<VisualCaptureFrames>() else {
        return;
    };
    frames.frames.extend(pending_frames);
}

fn drain_ready_offscreen_readbacks(
    pending_readbacks: &mut PendingOffscreenReadbacks,
    bridge: &OffscreenVisualCaptureBridge,
) {
    let ready_kinds = pending_readbacks
        .0
        .iter()
        .filter_map(|(kind, pending)| {
            let Ok(status) = pending.status.lock() else {
                return None;
            };
            status.as_ref().map(|_| *kind)
        })
        .collect::<Vec<_>>();

    for kind in ready_kinds {
        let Some(pending) = pending_readbacks.0.remove(&kind) else {
            continue;
        };
        let Ok(status) = pending.status.lock() else {
            continue;
        };
        if !matches!(status.as_ref(), Some(true)) {
            continue;
        }
        let buffer_slice = pending.buffer.slice(..);
        let padded_row_bytes = RenderDevice::align_copy_bytes_per_row(pending.width as usize * 4);
        let row_bytes = pending.width as usize * 4;
        let mapped = buffer_slice.get_mapped_range();
        let bytes = mapped
            .chunks(padded_row_bytes)
            .take(pending.height as usize)
            .flat_map(|row| row[..row_bytes.min(row.len())].iter().copied())
            .collect::<Vec<_>>();
        drop(mapped);
        pending.buffer.unmap();

        if let Ok(mut pending_frames) = bridge.0.lock() {
            pending_frames.insert(
                pending.key,
                CapturedVisualFrame {
                    width: pending.width,
                    height: pending.height,
                    bytes,
                    generation: pending.generation,
                },
            );
        }
    }
}
fn extract_offscreen_image_copiers(
    mut commands: Commands,
    image_copiers: Extract<Query<&OffscreenImageCopier>>,
) {
    commands.insert_resource(ExtractedOffscreenImageCopiers(
        image_copiers.iter().cloned().collect(),
    ));
}

fn extract_visual_capture_generation(
    mut commands: Commands,
    generation: Extract<Res<VisualCaptureGeneration>>,
) {
    commands.insert_resource(**generation);
}

impl bevy::render::render_graph::Node for OffscreenImageCopyNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(image_copiers) = world.get_resource::<ExtractedOffscreenImageCopiers>() else {
            return Ok(());
        };
        let Some(gpu_images) = world.get_resource::<RenderAssets<GpuImage>>() else {
            return Ok(());
        };
        let Some(render_queue) = world.get_resource::<RenderQueue>() else {
            return Ok(());
        };
        let pending_readbacks = world.get_resource::<PendingOffscreenReadbacks>();

        for image_copier in image_copiers.iter() {
            if pending_readbacks.is_some_and(|pending| pending.0.contains_key(&image_copier.key)) {
                continue;
            }
            let Some(src_image) = gpu_images.get(&image_copier.src_image) else {
                continue;
            };
            let mut encoder = render_context
                .render_device()
                .create_command_encoder(&CommandEncoderDescriptor::default());
            let padded_bytes_per_row =
                RenderDevice::align_copy_bytes_per_row(src_image.size.width as usize * 4);
            encoder.copy_texture_to_buffer(
                src_image.texture.as_image_copy(),
                TexelCopyBufferInfo {
                    buffer: &image_copier.buffer,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row as u32),
                        rows_per_image: None,
                    },
                },
                src_image.size,
            );
            render_queue.submit(std::iter::once(encoder.finish()));
        }

        Ok(())
    }
}

fn receive_offscreen_image_from_buffer(
    image_copiers: Res<ExtractedOffscreenImageCopiers>,
    render_device: Res<RenderDevice>,
    bridge: Res<OffscreenVisualCaptureBridge>,
    generation: Res<VisualCaptureGeneration>,
    mut pending_readbacks: ResMut<PendingOffscreenReadbacks>,
) {
    for image_copier in image_copiers.iter() {
        if pending_readbacks.0.contains_key(&image_copier.key) {
            continue;
        }
        let buffer_slice = image_copier.buffer.slice(..);
        let status = Arc::new(Mutex::new(None));
        let callback_status = status.clone();
        buffer_slice.map_async(MapMode::Read, move |result| {
            if let Ok(mut slot) = callback_status.lock() {
                *slot = Some(result.is_ok());
            }
        });
        pending_readbacks.0.insert(
            image_copier.key,
            PendingOffscreenReadback {
                key: image_copier.key,
                width: image_copier.width,
                height: image_copier.height,
                buffer: image_copier.buffer.clone(),
                generation: generation.0,
                status,
            },
        );
    }
    let _ = render_device.poll(PollType::Poll);
    drain_ready_offscreen_readbacks(&mut pending_readbacks, &bridge);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_capture_sessions_isolate_frames_by_session() {
        let mut sessions = SyntheticCaptureSessions::default();
        let key = VisualCaptureKey {
            kind: VisualSensorKind::Front,
            variant: VisualCaptureVariant::Rgb,
        };
        let session_a = sessions.begin_session();
        let session_b = sessions.begin_session();
        let request_a = sessions.register_request(session_a, key);
        let request_b = sessions.register_request(session_b, key);

        sessions.record_request_frame(
            request_a,
            CapturedVisualFrame {
                width: 4,
                height: 4,
                bytes: vec![1; 64],
                generation: 1,
            },
        );
        sessions.record_request_frame(
            request_b,
            CapturedVisualFrame {
                width: 4,
                height: 4,
                bytes: vec![2; 64],
                generation: 2,
            },
        );

        assert!(sessions.has_frame(session_a, key));
        assert!(sessions.has_frame(session_b, key));
        assert_eq!(
            sessions
                .take_frame(session_a, key)
                .map(|frame| frame.generation),
            Some(1)
        );
        assert_eq!(
            sessions
                .take_frame(session_b, key)
                .map(|frame| frame.generation),
            Some(2)
        );
    }

    #[test]
    fn synthetic_capture_sessions_clear_session_removes_only_its_state() {
        let mut sessions = SyntheticCaptureSessions::default();
        let key = VisualCaptureKey {
            kind: VisualSensorKind::Rear,
            variant: VisualCaptureVariant::Semantic,
        };
        let session_a = sessions.begin_session();
        let session_b = sessions.begin_session();
        let request_a = sessions.register_request(session_a, key);
        let request_b = sessions.register_request(session_b, key);

        sessions.record_request_frame(
            request_a,
            CapturedVisualFrame {
                width: 4,
                height: 4,
                bytes: vec![3; 64],
                generation: 3,
            },
        );
        sessions.record_request_frame(
            request_b,
            CapturedVisualFrame {
                width: 4,
                height: 4,
                bytes: vec![4; 64],
                generation: 4,
            },
        );

        sessions.clear_session(session_a);

        assert!(!sessions.has_frame(session_a, key));
        assert!(sessions.has_frame(session_b, key));
        assert_eq!(
            sessions
                .take_frame(session_b, key)
                .map(|frame| frame.generation),
            Some(4)
        );
    }
}
