use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    math::Affine2,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

use crate::core::config::{GroundAccentKind, GroundClutterKind, RepositoryConfig};

pub fn setup_scene(
    mut commands: Commands,
    config: Res<RepositoryConfig>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.insert_resource(ClearColor(Color::BLACK));

    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.79, 0.82, 0.88),
        brightness: 320.0,
        affects_lightmapped_meshes: true,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 18_500.0,
            shadows_enabled: true,
            color: Color::srgb(1.0, 0.96, 0.92),
            ..default()
        },
        Transform::from_xyz(1800.0, 2400.0, 900.0)
            .looking_at(Vec3::new(400.0, 200.0, 600.0), Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8_200.0,
            color: Color::srgb(0.58, 0.72, 0.9),
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-1500.0, 2200.0, -1800.0)
            .looking_at(Vec3::new(0.0, 500.0, 800.0), Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 4_600.0,
            color: Color::srgb(0.94, 0.78, 0.64),
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(900.0, 1500.0, -2200.0)
            .looking_at(Vec3::new(200.0, 250.0, 400.0), Vec3::Y),
    ));

    commands.spawn((
        PointLight {
            intensity: 1_200_000.0,
            range: 4200.0,
            radius: 18.0,
            color: Color::srgb(1.0, 0.72, 0.46),
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-300.0, 380.0, 700.0),
    ));

    commands.spawn((
        PointLight {
            intensity: 950_000.0,
            range: 3600.0,
            radius: 18.0,
            color: Color::srgb(0.62, 0.78, 1.0),
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(1150.0, 420.0, 1500.0),
    ));

    spawn_ground_tiles(&mut commands, &mut meshes, &mut materials, &mut images);

    spawn_scene_obstacles(&mut commands, &config, &mut meshes, &mut materials);
    spawn_ground_accents(&mut commands, &config, &mut meshes, &mut materials);
    spawn_ground_clutter(&mut commands, &config, &mut meshes, &mut materials);
    spawn_sky_markers(&mut commands, &config, &mut meshes, &mut materials);
}

fn spawn_ground_tiles(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
) {
    let ground_size = 20_000.0;
    let checker_cell_size = 1000.0;
    let checker_repeats = ground_size / checker_cell_size;
    let mut checker_texture = Image::new_fill(
        Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[
            61, 100, 56, 255, 44, 74, 41, 255, 44, 74, 41, 255, 61, 100, 56, 255,
        ],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    checker_texture.sampler = ImageSampler::Descriptor({
        let mut descriptor = ImageSamplerDescriptor::nearest();
        descriptor.set_address_mode(ImageAddressMode::Repeat);
        descriptor
    });
    let checker_texture = images.add(checker_texture);
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(checker_texture),
        perceptual_roughness: 0.98,
        uv_transform: Affine2::from_scale(Vec2::splat(checker_repeats)),
        ..default()
    });

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(ground_size, 2.0, ground_size))),
        MeshMaterial3d(ground_material),
        Transform::from_xyz(0.0, -1.0, 0.0),
    ));
}

fn spawn_scene_obstacles(
    commands: &mut Commands,
    config: &RepositoryConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    for obstacle in &config.scene.obstacles {
        let position = Vec3::from_array(obstacle.position);
        let scale = Vec3::from_array(obstacle.size);
        let color = Color::srgba(
            obstacle.color[0],
            obstacle.color[1],
            obstacle.color[2],
            obstacle.color[3],
        );
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(scale.x, scale.y, scale.z))),
            MeshMaterial3d(materials.add(color)),
            Transform::from_translation(position),
        ));
    }
}

fn spawn_ground_accents(
    commands: &mut Commands,
    config: &RepositoryConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    for accent in &config.scene.ground_accents {
        let position = Vec3::from_array(accent.position);
        let size = Vec3::from_array(accent.size);
        let color = Color::srgba(
            accent.color[0],
            accent.color[1],
            accent.color[2],
            accent.color[3],
        );

        match accent.kind {
            GroundAccentKind::SoilPatch => {
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::new(size.x, size.y, size.z))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: color,
                        perceptual_roughness: 1.0,
                        ..default()
                    })),
                    Transform::from_xyz(position.x, position.y + size.y * 0.5 + 0.02, position.z),
                ));
            }
            GroundAccentKind::RockCluster => {
                commands.spawn((
                    Mesh3d(meshes.add(Sphere::new(size.y * 0.55).mesh().uv(18, 12))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: color,
                        perceptual_roughness: 0.98,
                        ..default()
                    })),
                    Transform::from_xyz(position.x, position.y + size.y * 0.58, position.z)
                        .with_scale(Vec3::new(
                            size.x / size.y.max(1.0),
                            1.0,
                            size.z / size.y.max(1.0),
                        )),
                ));
            }
        }
    }
}

fn spawn_ground_clutter(
    commands: &mut Commands,
    config: &RepositoryConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    for clutter in &config.scene.ground_clutter {
        let position = Vec3::from_array(clutter.position);
        let color = Color::srgba(
            clutter.color[0],
            clutter.color[1],
            clutter.color[2],
            clutter.color[3],
        );

        match clutter.kind {
            GroundClutterKind::GrassPatch => {
                commands.spawn((
                    Mesh3d(meshes.add(Cylinder::new(clutter.footprint[0] * 0.42, 1.6))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: color,
                        perceptual_roughness: 1.0,
                        ..default()
                    })),
                    Transform::from_xyz(position.x, 1.0, position.z).with_scale(Vec3::new(
                        1.0,
                        1.0,
                        clutter.footprint[1] / clutter.footprint[0],
                    )),
                ));
            }
            GroundClutterKind::ShrubCluster => {
                commands.spawn((
                    Mesh3d(meshes.add(Sphere::new(clutter.height * 0.38).mesh().uv(18, 12))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: color,
                        perceptual_roughness: 0.96,
                        ..default()
                    })),
                    Transform::from_xyz(position.x, clutter.height * 0.38 + 0.06, position.z)
                        .with_scale(Vec3::new(
                            clutter.footprint[0] / clutter.height.max(1.0),
                            1.0,
                            clutter.footprint[1] / clutter.height.max(1.0),
                        )),
                ));
            }
            GroundClutterKind::TreeStand => {
                let trunk_height = clutter.height * 0.42;
                let crown_height = clutter.height * 0.78;
                commands.spawn((
                    Mesh3d(meshes.add(Cylinder::new(clutter.footprint[0] * 0.04, trunk_height))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(0.28, 0.20, 0.14),
                        perceptual_roughness: 0.98,
                        ..default()
                    })),
                    Transform::from_xyz(position.x, trunk_height * 0.5 + 0.04, position.z),
                ));
                commands.spawn((
                    Mesh3d(meshes.add(Sphere::new(crown_height * 0.26).mesh().uv(18, 12))),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: color,
                        perceptual_roughness: 0.98,
                        ..default()
                    })),
                    Transform::from_xyz(
                        position.x,
                        trunk_height + crown_height * 0.22 + 0.04,
                        position.z,
                    )
                    .with_scale(Vec3::new(
                        clutter.footprint[0] / crown_height.max(1.0),
                        1.0,
                        clutter.footprint[1] / crown_height.max(1.0),
                    )),
                ));
            }
        }
    }
}

fn spawn_sky_markers(
    commands: &mut Commands,
    config: &RepositoryConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    for marker in &config.scene.sky_markers {
        let position = Vec3::from_array(marker.position);
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(marker.radius).mesh().uv(18, 12))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgba(0.82, 0.92, 1.0, 0.45),
                emissive: Color::srgb(0.08, 0.12, 0.14).into(),
                perceptual_roughness: 0.9,
                metallic: 0.0,
                ..default()
            })),
            Transform::from_translation(position),
        ));
    }
}
