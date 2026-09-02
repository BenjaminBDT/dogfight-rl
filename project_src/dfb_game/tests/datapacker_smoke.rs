use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use dfb_game::api::types::{
    AircraftObservation, ArenaObservation, AudioObservation, EnvironmentAction,
    ObservationCaptureConfig, PixelFormat, StateObservation, VisualResolutionMode,
    VisualSensorConfig, VisualSensorKind,
};
use dfb_game::dataset_tool::label::{
    DerivedLabelsManifest, DerivedStepLabels, SparseVotingArtifactRef,
};
use dfb_game::policy_contract::{ACTION_SCHEMA_ID, POLICY_CONTRACT_ID};
use dfb_game::recording::{
    AudioArtifactMetadata, AudioArtifactRef, DerivedArtifactConvention, DerivedEpisodeManifest,
    DerivedStepArtifacts, InitialWorldSnapshot, RecordedDynamicWorldState, RecordedEpisodeManifest,
    RecordedStep, RecordedStepArtifacts, RecordedStepChunk, RecordedStepChunkIndexEntry,
    VisualArtifactRef,
};
use ndarray::{Ix1, Ix2, Ix3, Ix4};
use ndarray_npy::NpzReader;

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{unique}"))
}

fn write_ron<T: serde::Serialize>(path: &Path, value: &T) {
    let payload = ron::ser::to_string_pretty(value, ron::ser::PrettyConfig::default()).unwrap();
    fs::write(path, payload).unwrap();
}

fn aircraft_observations() -> Vec<AircraftObservation> {
    vec![
        AircraftObservation {
            role: "fighter1".to_string(),
            position: [0.0, 0.0, 0.0],
            orientation_quat: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0, 0.0, 0.0],
            angular_velocity_deg: [0.0, 0.0, 0.0],
            forward: [0.0, 0.0, 1.0],
            ..Default::default()
        },
        AircraftObservation {
            role: "fighter2".to_string(),
            position: [100.0, 0.0, 0.0],
            orientation_quat: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0, 0.0, 0.0],
            angular_velocity_deg: [0.0, 0.0, 0.0],
            forward: [0.0, 0.0, 1.0],
            ..Default::default()
        },
    ]
}

#[test]
fn datapacker_accepts_minimal_authoritative_episode_fixture() {
    let visual_width = 16_u32;
    let visual_height = 16_u32;
    let root = unique_temp_dir("dfb-datapacker-smoke");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join(".dfb_dev_root"), "test workspace root\n").unwrap();
    let episode_root = root
        .join("datasets")
        .join("dfb_game")
        .join("recordings")
        .join("episode_smoke");
    let output_root = root.join("dataset_out");
    fs::create_dir_all(episode_root.join("steps")).unwrap();
    fs::create_dir_all(episode_root.join("derived/fighter1/visual/front")).unwrap();
    fs::create_dir_all(episode_root.join("derived/fighter1/visual/rear")).unwrap();
    fs::create_dir_all(episode_root.join("derived/fighter1/segmentation/front")).unwrap();
    fs::create_dir_all(episode_root.join("derived/fighter1/segmentation/rear")).unwrap();
    fs::create_dir_all(episode_root.join("derived/fighter1/audio")).unwrap();

    let initial = InitialWorldSnapshot {
        state: StateObservation {
            tick: 0,
            seed: 42,
            sim_time_seconds: 0.0,
            match_phase: "Running".to_string(),
            scene_name: "default".to_string(),
            aircraft: aircraft_observations(),
            arena: ArenaObservation::default(),
            ..Default::default()
        },
        dynamic: RecordedDynamicWorldState::default(),
        audio_semantics: None,
    };
    write_ron(&episode_root.join("initial_state.ron"), &initial);

    let step = RecordedStep {
        index: 0,
        tick: 1,
        sim_time_seconds: 1.0 / 60.0,
        fighter1_command: EnvironmentAction::default(),
        fighter2_command: EnvironmentAction::default(),
        state: StateObservation {
            tick: 1,
            seed: 42,
            sim_time_seconds: 1.0 / 60.0,
            match_phase: "Finished".to_string(),
            scene_name: "default".to_string(),
            aircraft: aircraft_observations(),
            arena: ArenaObservation::default(),
            ..Default::default()
        },
        dynamic: RecordedDynamicWorldState::default(),
        audio_semantics: None,
    };
    write_ron(
        &episode_root.join("steps/chunk_000000.ron"),
        &RecordedStepChunk {
            chunk_index: 0,
            start_step_index: 0,
            steps: vec![step],
        },
    );

    let manifest = RecordedEpisodeManifest {
        schema_version: 12,
        policy_contract_id: POLICY_CONTRACT_ID.to_string(),
        action_schema_id: ACTION_SCHEMA_ID.to_string(),
        episode_id: "episode_smoke".to_string(),
        scene_name: "default".to_string(),
        seed: Some(42),
        fixed_time_step_seconds: 1.0 / 60.0,
        capture_config: ObservationCaptureConfig::default(),
        audio_artifact_metadata: Some(AudioArtifactMetadata {
            sample_rate: 48_000,
            channels: 2,
            window_seconds: 1.0 / 60.0,
        }),
        started_tick: 0,
        started_sim_time_seconds: 0.0,
        final_tick: 1,
        final_sim_time_seconds: 1.0 / 60.0,
        total_steps: 1,
        step_chunks: vec![RecordedStepChunkIndexEntry {
            chunk_index: 0,
            start_step_index: 0,
            step_count: 1,
            file_path: "steps/chunk_000000.ron".to_string(),
        }],
        step_artifacts: vec![RecordedStepArtifacts {
            index: 0,
            tick: 1,
            visual: Vec::new(),
            audio: None,
        }],
        termination_reason: "match_finished".to_string(),
        winner: Some("fighter1".to_string()),
        bridge_role: Some("Server".to_string()),
        bridge_session: Some("smoke".to_string()),
        bridge_transport: Some("InProcess".to_string()),
        authoritative_source: true,
        initial_state_path: "initial_state.ron".to_string(),
        artifact_convention: Default::default(),
    };
    write_ron(&episode_root.join("episode.ron"), &manifest);

    let ppm_bytes = {
        let mut bytes = format!("P6\n{} {}\n255\n", visual_width, visual_height).into_bytes();
        for _ in 0..(visual_width * visual_height) {
            bytes.extend_from_slice(&[255, 0, 0]);
        }
        bytes
    };
    fs::write(
        episode_root.join("derived/fighter1/visual/front/step_000000.ppm"),
        ppm_bytes,
    )
    .unwrap();
    let ppm_bytes = {
        let mut bytes = format!("P6\n{} {}\n255\n", visual_width, visual_height).into_bytes();
        for _ in 0..(visual_width * visual_height) {
            bytes.extend_from_slice(&[0, 255, 0]);
        }
        bytes
    };
    fs::write(
        episode_root.join("derived/fighter1/visual/rear/step_000000.ppm"),
        ppm_bytes,
    )
    .unwrap();
    let segmentation_bytes = {
        let mut bytes = format!("P5\n{} {}\n255\n", visual_width, visual_height).into_bytes();
        bytes.extend(std::iter::repeat_n(
            1_u8,
            (visual_width * visual_height) as usize,
        ));
        bytes
    };
    fs::write(
        episode_root.join("derived/fighter1/segmentation/front/step_000000.pgm"),
        segmentation_bytes,
    )
    .unwrap();
    let segmentation_bytes = {
        let mut bytes = format!("P5\n{} {}\n255\n", visual_width, visual_height).into_bytes();
        bytes.extend(std::iter::repeat_n(
            2_u8,
            (visual_width * visual_height) as usize,
        ));
        bytes
    };
    fs::write(
        episode_root.join("derived/fighter1/segmentation/rear/step_000000.pgm"),
        segmentation_bytes,
    )
    .unwrap();
    let audio = AudioObservation {
        sample_rate: 48_000,
        channels: 2,
        window_seconds: 1.0 / 60.0,
        samples: vec![0.0; 2 * 800],
        ..Default::default()
    };
    let audio_bytes = encode_wav_i16_bytes(audio.sample_rate, audio.channels, &audio.samples);
    fs::write(
        episode_root.join("derived/fighter1/audio/step_000000.wav"),
        audio_bytes,
    )
    .unwrap();
    let capture_config = ObservationCaptureConfig {
        enable_visual: true,
        enable_audio: true,
        visual_sensors: vec![
            VisualSensorConfig {
                kind: VisualSensorKind::Front,
                width: visual_width,
                height: visual_height,
                format: PixelFormat::Rgb8,
                resolution_mode: VisualResolutionMode::Fixed,
                include_hud: false,
                capture_variants: Vec::new(),
            },
            VisualSensorConfig {
                kind: VisualSensorKind::Rear,
                width: visual_width,
                height: visual_height,
                format: PixelFormat::Rgb8,
                resolution_mode: VisualResolutionMode::Fixed,
                include_hud: false,
                capture_variants: Vec::new(),
            },
        ],
        audio_window_seconds: 1.0 / 60.0,
    };
    let derived_manifest = DerivedEpisodeManifest {
        schema_version: 2,
        source_episode_id: "episode_smoke".to_string(),
        source_episode_root: episode_root.display().to_string(),
        observed_role: "fighter1".to_string(),
        capture_config,
        audio_artifact_metadata: Some(AudioArtifactMetadata {
            sample_rate: 48_000,
            channels: 2,
            window_seconds: 1.0 / 60.0,
        }),
        initial_tick: 0,
        total_steps: 1,
        initial_visual: Vec::new(),
        initial_segmentation: Vec::new(),
        initial_audio: None,
        steps: vec![DerivedStepArtifacts {
            index: 0,
            tick: 1,
            visual: vec![
                VisualArtifactRef {
                    camera: VisualSensorKind::Front,
                    file_path: Some("visual/front/step_000000.ppm".to_string()),
                    width: Some(visual_width),
                    height: Some(visual_height),
                    format: Some(PixelFormat::Rgb8),
                    byte_offset: None,
                    byte_length: None,
                },
                VisualArtifactRef {
                    camera: VisualSensorKind::Rear,
                    file_path: Some("visual/rear/step_000000.ppm".to_string()),
                    width: Some(visual_width),
                    height: Some(visual_height),
                    format: Some(PixelFormat::Rgb8),
                    byte_offset: None,
                    byte_length: None,
                },
            ],
            segmentation: vec![
                VisualArtifactRef {
                    camera: VisualSensorKind::Front,
                    file_path: Some("segmentation/front/step_000000.pgm".to_string()),
                    width: Some(visual_width),
                    height: Some(visual_height),
                    format: Some(PixelFormat::Gray8),
                    byte_offset: None,
                    byte_length: None,
                },
                VisualArtifactRef {
                    camera: VisualSensorKind::Rear,
                    file_path: Some("segmentation/rear/step_000000.pgm".to_string()),
                    width: Some(visual_width),
                    height: Some(visual_height),
                    format: Some(PixelFormat::Gray8),
                    byte_offset: None,
                    byte_length: None,
                },
            ],
            audio: Some(AudioArtifactRef {
                file_path: Some("audio/step_000000.wav".to_string()),
                byte_offset: None,
                byte_length: None,
            }),
        }],
        artifact_convention: DerivedArtifactConvention::default(),
    };
    write_ron(
        &episode_root.join("derived/fighter1/derived_modalities.ron"),
        &derived_manifest,
    );
    let labels = DerivedLabelsManifest {
        schema_version: 1,
        source_episode_id: "episode_smoke".to_string(),
        source_episode_root: episode_root.display().to_string(),
        observed_role: "fighter1".to_string(),
        derived_manifest_path: episode_root
            .join("derived/fighter1/derived_modalities.ron")
            .display()
            .to_string(),
        coordinate_convention_id: "dfb_aircraft_body_rhs_forward_pos_z_v1".to_string(),
        keypoint_schema_id: "fighter_surface_fps8_plus_center_v1".to_string(),
        audio_cue_schema_id: "binaural_cue_schema_v1".to_string(),
        visual_label_mode: "projected_keypoints_plus_segmentation_visibility_v1".to_string(),
        notes: Vec::new(),
        steps: vec![DerivedStepLabels {
            index: 0,
            tick: 1,
            sim_time_seconds: 1.0 / 60.0,
            gt_relative_position_body: [0.0, 0.0, 0.0],
            gt_doa_unit_vector_body: [0.0, 0.0, 1.0],
            gt_log_distance_scalar: 0.0,
            gt_relative_orientation_quat: [0.0, 0.0, 0.0, 1.0],
            gt_relative_linear_velocity_body: [0.0, 0.0, 0.0],
            gt_relative_angular_velocity_body: [0.0, 0.0, 0.0],
            keypoints_2d_front: vec![[4.0, 5.0]; 9],
            keypoints_2d_rear: vec![[6.0, 7.0]; 9],
            keypoint_visibility_front: vec![1; 9],
            keypoint_visibility_rear: vec![1; 9],
            keypoint_projectable_front: vec![1; 9],
            keypoint_projectable_rear: vec![1; 9],
            keypoint_voting_front: SparseVotingArtifactRef {
                file_path: "vision_voting/front.bin".to_string(),
                byte_offset: 0,
                byte_length: 0,
                width: visual_width,
                height: visual_height,
                keypoint_count: 9,
                pixel_count: 0,
                coord_dtype: "u16".to_string(),
                vector_dtype: "float16".to_string(),
            },
            keypoint_voting_rear: SparseVotingArtifactRef {
                file_path: "vision_voting/rear.bin".to_string(),
                byte_offset: 0,
                byte_length: 0,
                width: visual_width,
                height: visual_height,
                keypoint_count: 9,
                pixel_count: 0,
                coord_dtype: "u16".to_string(),
                vector_dtype: "float16".to_string(),
            },
            binaural_energy_t: [0.0; 4],
            binaural_cue_vector_t: [0.0; 10],
            target_pos_conf: 1.0,
            target_ori_conf: 1.0,
        }],
    };
    write_ron(
        &episode_root.join("derived/fighter1/derived_labels.ron"),
        &labels,
    );
    fs::create_dir_all(episode_root.join("derived/fighter1/vision_voting")).unwrap();
    fs::write(
        episode_root.join("derived/fighter1/vision_voting/front.bin"),
        [],
    )
    .unwrap();
    fs::write(
        episode_root.join("derived/fighter1/vision_voting/rear.bin"),
        [],
    )
    .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_dfb_tool_dataset"))
        .arg("pack")
        .arg("--episode")
        .arg(&episode_root)
        .arg("--observed-role")
        .arg("fighter1")
        .arg("--output-dir")
        .arg(&output_root)
        .arg("--force")
        .status()
        .unwrap();
    assert!(status.success());

    assert!(output_root.join("schema.json").exists());
    assert!(output_root.join("meta.json").exists());
    assert!(output_root.join("core/chunk_000000.npz").exists());
    assert!(output_root.join("vision_labels/chunk_000000.npz").exists());
    assert!(output_root.join("audio_features/chunk_000000.npz").exists());
    assert!(output_root.join("rule_targets/chunk_000000.npz").exists());

    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(output_root.join("meta.json")).unwrap()).unwrap();
    assert_eq!(
        meta["schema_id"].as_str(),
        Some("dfb_state_estimation_dataset_schema_v2")
    );
    assert_eq!(
        meta["storage_layout"]["format"].as_str(),
        Some("chunked_npz")
    );
    assert_eq!(meta["episodes"].as_array().map(Vec::len), Some(1));
    assert_eq!(meta["chunks"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        meta["statistics"]["total_simulation_steps"].as_u64(),
        Some(1)
    );

    let mut core =
        NpzReader::new(File::open(output_root.join("core/chunk_000000.npz")).unwrap()).unwrap();
    let simulation_step_index = core
        .by_name::<ndarray::OwnedRepr<i32>, Ix1>("simulation_step_index")
        .unwrap();
    let timestamp = core
        .by_name::<ndarray::OwnedRepr<f64>, Ix1>("timestamp")
        .unwrap();
    let front_camera_image = core
        .by_name::<ndarray::OwnedRepr<u8>, Ix4>("front_camera_image")
        .unwrap();
    let audio_window = core
        .by_name::<ndarray::OwnedRepr<f32>, Ix3>("audio_window_binaural")
        .unwrap();
    let gt_relative_position = core
        .by_name::<ndarray::OwnedRepr<f32>, Ix2>("gt_relative_position")
        .unwrap();
    assert_eq!(simulation_step_index.shape(), &[1]);
    assert_eq!(simulation_step_index[[0]], 0);
    assert_eq!(timestamp.shape(), &[1]);
    assert_eq!(
        front_camera_image.shape(),
        &[1, visual_height as usize, visual_width as usize, 4]
    );
    assert_eq!(audio_window.shape(), &[1, 800, 2]);
    assert_eq!(gt_relative_position.shape(), &[1, 3]);

    let mut vision =
        NpzReader::new(File::open(output_root.join("vision_labels/chunk_000000.npz")).unwrap())
            .unwrap();
    let segmentation_front = vision
        .by_name::<ndarray::OwnedRepr<u8>, Ix3>("segmentation_mask_front")
        .unwrap();
    let keypoints_front = vision
        .by_name::<ndarray::OwnedRepr<f32>, Ix3>("keypoints_2d_front")
        .unwrap();
    let visibility_front = vision
        .by_name::<ndarray::OwnedRepr<u8>, Ix2>("keypoint_visibility_front")
        .unwrap();
    let voting_pixels_front = vision
        .by_name::<ndarray::OwnedRepr<u16>, Ix3>("keypoint_voting_pixels_front")
        .unwrap();
    let voting_vectors_front = vision
        .by_name::<ndarray::OwnedRepr<f32>, Ix4>("keypoint_voting_unit_vectors_front")
        .unwrap();
    let voting_mask_front = vision
        .by_name::<ndarray::OwnedRepr<u8>, Ix2>("keypoint_voting_mask_front")
        .unwrap();
    assert_eq!(
        segmentation_front.shape(),
        &[1, visual_height as usize, visual_width as usize]
    );
    assert_eq!(keypoints_front.shape(), &[1, 9, 2]);
    assert_eq!(keypoints_front[[0, 0, 0]], 4.0);
    assert_eq!(keypoints_front[[0, 0, 1]], 5.0);
    assert_eq!(visibility_front.shape(), &[1, 9]);
    assert_eq!(visibility_front[[0, 0]], 1);
    assert_eq!(voting_pixels_front.shape(), &[1, 0, 2]);
    assert_eq!(voting_vectors_front.shape(), &[1, 0, 9, 2]);
    assert_eq!(voting_mask_front.shape(), &[1, 0]);
    assert!(
        segmentation_front.iter().any(|value| *value != 0),
        "segmentation mask should contain at least one non-background pixel"
    );

    let mut audio =
        NpzReader::new(File::open(output_root.join("audio_features/chunk_000000.npz")).unwrap())
            .unwrap();
    let binaural_energy_t = audio
        .by_name::<ndarray::OwnedRepr<f32>, Ix2>("binaural_energy_t")
        .unwrap();
    let binaural_cue_vector_t = audio
        .by_name::<ndarray::OwnedRepr<f32>, Ix2>("binaural_cue_vector_t")
        .unwrap();
    assert_eq!(binaural_energy_t.shape(), &[1, 4]);
    assert_eq!(binaural_cue_vector_t.shape(), &[1, 10]);
    assert!(
        audio
            .by_name::<ndarray::OwnedRepr<f32>, Ix2>("channel_energy")
            .is_err(),
        "legacy 7.1 channel_energy field should no longer be exported"
    );
    assert!(
        audio
            .by_name::<ndarray::OwnedRepr<f32>, Ix2>("directional_audio_vector")
            .is_err(),
        "legacy directional_audio_vector field should no longer be exported"
    );
    assert!(
        audio
            .by_name::<ndarray::OwnedRepr<f32>, Ix2>("delta_audio_vector")
            .is_err(),
        "delta_binaural_cue_t is runtime-assembled and should not be persisted as a step field"
    );

    let mut rules =
        NpzReader::new(File::open(output_root.join("rule_targets/chunk_000000.npz")).unwrap())
            .unwrap();
    let target_pos_conf = rules
        .by_name::<ndarray::OwnedRepr<f32>, Ix1>("target_pos_conf")
        .unwrap();
    let target_ori_conf = rules
        .by_name::<ndarray::OwnedRepr<f32>, Ix1>("target_ori_conf")
        .unwrap();
    assert_eq!(target_pos_conf.shape(), &[1]);
    assert_eq!(target_pos_conf[[0]], 1.0);
    assert_eq!(target_ori_conf.shape(), &[1]);
    assert_eq!(target_ori_conf[[0]], 1.0);

    fs::remove_dir_all(root).ok();
}

fn encode_wav_i16_bytes(sample_rate: u32, channels: u16, samples: &[f32]) -> Vec<u8> {
    let bytes_per_sample = 2u16;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * block_align as u32;
    let data_len = (samples.len() * bytes_per_sample as usize) as u32;
    let riff_len = 36 + data_len;
    let mut bytes = Vec::with_capacity((44 + data_len) as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_len.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&(bytes_per_sample * 8).to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    bytes
}
