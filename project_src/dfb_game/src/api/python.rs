use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;

use crate::api::commands::TargetedEnvironmentAction;
use crate::api::environment::EnvironmentInstance;
use crate::api::types::{
    AudioObservation, EnvironmentAction, EnvironmentAgentControlConfig, EnvironmentAgentMode,
    EnvironmentResetOptions, ObservationBundle, ObservationCaptureConfig, PixelFormat,
    VisualCaptureVariant, VisualObservation, VisualResolutionMode, VisualSensorConfig,
    VisualSensorKind,
};
use crate::core::config::{ConfigPaths, resolve_project_root};
use crate::recording::reconstruct::{RecordingAccess, RecordingReconstructionSession};
use crate::simulation::components::{AircraftRole, ControlAuthority};

fn py_runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(py_runtime_error)
}

fn parse_json_arg<T>(name: &str, json: Option<String>) -> PyResult<T>
where
    T: for<'de> Deserialize<'de> + Default,
{
    match json {
        Some(json) => serde_json::from_str(&json)
            .map_err(|error| PyValueError::new_err(format!("invalid {name} JSON: {error}"))),
        None => Ok(T::default()),
    }
}

fn parse_visual_sensor_kind(value: &str) -> PyResult<VisualSensorKind> {
    match value {
        "front" | "Front" => Ok(VisualSensorKind::Front),
        "rear" | "Rear" => Ok(VisualSensorKind::Rear),
        _ => Err(PyValueError::new_err(format!(
            "invalid visual sensor kind: {value}"
        ))),
    }
}

fn parse_pixel_format(value: &str) -> PyResult<PixelFormat> {
    match value {
        "rgb8" | "Rgb8" => Ok(PixelFormat::Rgb8),
        "rgba8" | "Rgba8" => Ok(PixelFormat::Rgba8),
        "gray8" | "Gray8" => Ok(PixelFormat::Gray8),
        _ => Err(PyValueError::new_err(format!(
            "invalid pixel format: {value}"
        ))),
    }
}

fn parse_visual_resolution_mode(value: &str) -> PyResult<VisualResolutionMode> {
    match value {
        "fixed" | "Fixed" => Ok(VisualResolutionMode::Fixed),
        "runtime_window" | "RuntimeWindow" => Ok(VisualResolutionMode::RuntimeWindow),
        _ => Err(PyValueError::new_err(format!(
            "invalid visual resolution mode: {value}"
        ))),
    }
}

fn parse_agent_mode(value: &str) -> PyResult<EnvironmentAgentMode> {
    match value {
        "external" | "External" => Ok(EnvironmentAgentMode::External),
        "model" | "Model" => Ok(EnvironmentAgentMode::Model),
        "built_in_ai" | "BuiltInAi" => Ok(EnvironmentAgentMode::BuiltInAi),
        "built_in_ai_precise" | "BuiltInAiPrecise" => Ok(EnvironmentAgentMode::BuiltInAiPrecise),
        "built_in_ai_imperfect" | "BuiltInAiImperfect" => {
            Ok(EnvironmentAgentMode::BuiltInAiImperfect)
        }
        "built_in_ai_teacher" | "BuiltInAiTeacher" => Ok(EnvironmentAgentMode::BuiltInAiTeacher),
        "built_in_ai_passive_bounce" | "BuiltInAiPassiveBounce" => {
            Ok(EnvironmentAgentMode::BuiltInAiPassiveBounce)
        }
        _ => Err(PyValueError::new_err(format!(
            "invalid environment agent mode: {value}"
        ))),
    }
}

fn parse_aircraft_role(value: &str) -> PyResult<AircraftRole> {
    match value {
        "fighter1" | "Fighter1" => Ok(AircraftRole::Fighter1),
        "fighter2" | "Fighter2" => Ok(AircraftRole::Fighter2),
        _ => Err(PyValueError::new_err(format!(
            "invalid aircraft role: {value}"
        ))),
    }
}

fn parse_control_authority(value: &str) -> PyResult<ControlAuthority> {
    match value {
        "human" | "Human" => Ok(ControlAuthority::Human),
        "built_in_ai" | "BuiltInAi" => Ok(ControlAuthority::BuiltInAi),
        "external_agent" | "ExternalAgent" => Ok(ControlAuthority::ExternalAgent),
        "replay" | "Replay" => Ok(ControlAuthority::Replay),
        _ => Err(PyValueError::new_err(format!(
            "invalid control authority: {value}"
        ))),
    }
}

#[derive(Debug, Clone, Deserialize)]
struct JsonVisualSensorConfig {
    kind: String,
    width: u32,
    height: u32,
    format: String,
    resolution_mode: String,
    include_hud: bool,
    #[serde(default)]
    capture_variants: Vec<String>,
}

fn parse_visual_capture_variant(value: &str) -> PyResult<VisualCaptureVariant> {
    match value {
        "rgb" | "Rgb" | "RGB" => Ok(VisualCaptureVariant::Rgb),
        "semantic" | "Semantic" => Ok(VisualCaptureVariant::Semantic),
        _ => Err(PyValueError::new_err(format!(
            "invalid visual capture variant: {value}"
        ))),
    }
}

impl TryFrom<JsonVisualSensorConfig> for VisualSensorConfig {
    type Error = PyErr;

    fn try_from(value: JsonVisualSensorConfig) -> Result<Self, Self::Error> {
        Ok(Self {
            kind: parse_visual_sensor_kind(&value.kind)?,
            width: value.width,
            height: value.height,
            format: parse_pixel_format(&value.format)?,
            resolution_mode: parse_visual_resolution_mode(&value.resolution_mode)?,
            include_hud: value.include_hud,
            capture_variants: value
                .capture_variants
                .into_iter()
                .map(|variant| parse_visual_capture_variant(&variant))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct JsonTargetedEnvironmentAction {
    role: String,
    action: EnvironmentAction,
}

impl TryFrom<JsonTargetedEnvironmentAction> for TargetedEnvironmentAction {
    type Error = PyErr;

    fn try_from(value: JsonTargetedEnvironmentAction) -> Result<Self, Self::Error> {
        Ok(Self {
            role: parse_aircraft_role(&value.role)?,
            action: value.action,
        })
    }
}

fn parse_visual_sensors_json(json: Option<String>) -> PyResult<Vec<VisualSensorConfig>> {
    parse_json_arg::<Vec<JsonVisualSensorConfig>>("visual_sensors", json)?
        .into_iter()
        .map(TryInto::try_into)
        .collect()
}

fn visual_kind_matches(kind: VisualSensorKind, requested: &str) -> bool {
    matches!(
        (kind, requested),
        (VisualSensorKind::Front, "front" | "Front") | (VisualSensorKind::Rear, "rear" | "Rear")
    )
}

fn find_visual_frame<'a>(
    observation: &'a ObservationBundle,
    camera: &str,
) -> PyResult<&'a VisualObservation> {
    observation
        .visual
        .iter()
        .find(|frame| visual_kind_matches(frame.camera, camera))
        .ok_or_else(|| PyValueError::new_err(format!("missing visual frame for camera: {camera}")))
}

fn find_audio_observation(observation: &ObservationBundle) -> PyResult<&AudioObservation> {
    observation
        .audio
        .as_ref()
        .ok_or_else(|| PyValueError::new_err("missing audio observation"))
}

fn encode_f32_samples_le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * std::mem::size_of::<f32>());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn build_capture_config(
    enable_visual: bool,
    enable_audio: bool,
    visual_sensors_json: Option<String>,
    audio_window_seconds: f32,
) -> PyResult<ObservationCaptureConfig> {
    if audio_window_seconds <= 0.0 {
        return Err(PyValueError::new_err(
            "audio_window_seconds must be positive",
        ));
    }

    Ok(ObservationCaptureConfig {
        enable_visual,
        enable_audio,
        visual_sensors: parse_visual_sensors_json(visual_sensors_json)?,
        audio_window_seconds,
    })
}

fn build_config_paths(
    project_root: Option<String>,
    scene_name: Option<&str>,
    scene_path: Option<&str>,
) -> ConfigPaths {
    ConfigPaths {
        project_root: project_root
            .map(PathBuf::from)
            .unwrap_or_else(resolve_project_root),
        scene_override: scene_name.map(ToOwned::to_owned),
        scene_override_path: scene_path.map(PathBuf::from),
    }
}

fn build_reset_options(
    scene_name: Option<String>,
    scene_path: Option<String>,
    seed: Option<u64>,
    enable_visual: bool,
    enable_audio: bool,
    visual_sensors_json: Option<String>,
    audio_window_seconds: f32,
    ticks_per_step: u32,
    self_play: bool,
    fighter1_mode: Option<String>,
    fighter2_mode: Option<String>,
) -> PyResult<EnvironmentResetOptions> {
    if scene_name.is_some() && scene_path.is_some() {
        return Err(PyValueError::new_err(
            "scene_name and scene_path cannot be provided together",
        ));
    }
    if audio_window_seconds <= 0.0 {
        return Err(PyValueError::new_err(
            "audio_window_seconds must be positive",
        ));
    }
    if ticks_per_step == 0 {
        return Err(PyValueError::new_err("ticks_per_step must be at least 1"));
    }

    let visual_sensors = parse_visual_sensors_json(visual_sensors_json)?;
    let agent_control = match (fighter1_mode, fighter2_mode) {
        (Some(fighter1), Some(fighter2)) => EnvironmentAgentControlConfig {
            fighter1: parse_agent_mode(&fighter1)?,
            fighter2: parse_agent_mode(&fighter2)?,
        },
        (None, None) => {
            if self_play {
                EnvironmentAgentControlConfig::self_play()
            } else {
                EnvironmentAgentControlConfig::single_agent_vs_ai()
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "fighter1_mode and fighter2_mode must be both provided or both omitted",
            ));
        }
    };

    Ok(EnvironmentResetOptions {
        scene_name,
        scene_path,
        seed,
        enable_visual,
        enable_audio,
        visual_sensors,
        audio_window_seconds,
        ticks_per_step,
        agent_control,
    })
}

#[pyclass(name = "EnvironmentAction")]
#[derive(Clone)]
pub struct PyEnvironmentAction {
    inner: EnvironmentAction,
}

#[pymethods]
impl PyEnvironmentAction {
    #[new]
    #[pyo3(signature = (
        throttle=0.0,
        brake=false,
        pitch=0.0,
        roll=0.0,
        yaw=0.0,
        fire_gun=false,
        repair=false
    ))]
    fn new(
        throttle: f32,
        brake: bool,
        pitch: f32,
        roll: f32,
        yaw: f32,
        fire_gun: bool,
        repair: bool,
    ) -> Self {
        Self {
            inner: EnvironmentAction {
                throttle,
                brake,
                pitch,
                roll,
                yaw,
                fire_gun,
                repair,
            },
        }
    }

    fn json(&self) -> PyResult<String> {
        to_json(&self.inner)
    }
}

#[pyclass(name = "Environment", unsendable)]
pub struct PyEnvironment {
    inner: EnvironmentInstance,
}

#[pyclass(name = "EpisodeRecording")]
pub struct PyEpisodeRecording {
    inner: RecordingAccess,
}

#[pyclass(name = "EpisodeReconstructor", unsendable)]
pub struct PyEpisodeReconstructor {
    inner: RecordingReconstructionSession,
}

#[pymethods]
impl PyEnvironment {
    #[new]
    #[pyo3(signature = (
        project_root=None,
        scene_name=None,
        scene_path=None,
        seed=None,
        enable_visual=false,
        enable_audio=false,
        visual_sensors_json=None,
        audio_window_seconds=0.25,
        ticks_per_step=1,
        self_play=false,
        fighter1_mode=None,
        fighter2_mode=None
    ))]
    fn new(
        project_root: Option<String>,
        scene_name: Option<String>,
        scene_path: Option<String>,
        seed: Option<u64>,
        enable_visual: bool,
        enable_audio: bool,
        visual_sensors_json: Option<String>,
        audio_window_seconds: f32,
        ticks_per_step: u32,
        self_play: bool,
        fighter1_mode: Option<String>,
        fighter2_mode: Option<String>,
    ) -> PyResult<Self> {
        let options = build_reset_options(
            scene_name.clone(),
            scene_path.clone(),
            seed,
            enable_visual,
            enable_audio,
            visual_sensors_json,
            audio_window_seconds,
            ticks_per_step,
            self_play,
            fighter1_mode,
            fighter2_mode,
        )?;
        Ok(Self {
            inner: EnvironmentInstance::new_headless(
                build_config_paths(project_root, scene_name.as_deref(), scene_path.as_deref()),
                options,
            ),
        })
    }

    #[pyo3(signature = (
        scene_name=None,
        scene_path=None,
        seed=None,
        enable_visual=false,
        enable_audio=false,
        visual_sensors_json=None,
        audio_window_seconds=0.25,
        ticks_per_step=1,
        self_play=false,
        fighter1_mode=None,
        fighter2_mode=None
    ))]
    fn reset_json(
        &mut self,
        scene_name: Option<String>,
        scene_path: Option<String>,
        seed: Option<u64>,
        enable_visual: bool,
        enable_audio: bool,
        visual_sensors_json: Option<String>,
        audio_window_seconds: f32,
        ticks_per_step: u32,
        self_play: bool,
        fighter1_mode: Option<String>,
        fighter2_mode: Option<String>,
    ) -> PyResult<String> {
        let options = build_reset_options(
            scene_name,
            scene_path,
            seed,
            enable_visual,
            enable_audio,
            visual_sensors_json,
            audio_window_seconds,
            ticks_per_step,
            self_play,
            fighter1_mode,
            fighter2_mode,
        )?;
        to_json(&self.inner.reset(&options))
    }

    fn latest_observation_json(&mut self) -> PyResult<String> {
        to_json(&self.inner.latest_observation())
    }

    fn latest_visual_bytes<'py>(
        &mut self,
        py: Python<'py>,
        camera: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let observation = self.inner.latest_observation();
        let frame = find_visual_frame(&observation, &camera)?;
        Ok(PyBytes::new(py, &frame.bytes))
    }

    fn latest_audio_samples_bytes<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let observation = self.inner.latest_observation();
        let audio = find_audio_observation(&observation)?;
        Ok(PyBytes::new(py, &encode_f32_samples_le(&audio.samples)))
    }

    fn episode_status_json(&mut self) -> PyResult<String> {
        to_json(&self.inner.episode_status())
    }

    fn recording_status_json(&self) -> PyResult<String> {
        to_json(&self.inner.recording_status())
    }

    fn step_json(&mut self, command: &PyEnvironmentAction) -> PyResult<String> {
        to_json(&self.inner.step(command.inner))
    }

    fn step_self_play_json(
        &mut self,
        fighter1_command: &PyEnvironmentAction,
        fighter2_command: &PyEnvironmentAction,
    ) -> PyResult<String> {
        to_json(
            &self
                .inner
                .step_self_play(fighter1_command.inner, fighter2_command.inner),
        )
    }

    #[pyo3(signature = (actions_json))]
    fn step_targeted_json(&mut self, actions_json: String) -> PyResult<String> {
        let actions = parse_json_arg::<Vec<JsonTargetedEnvironmentAction>>(
            "targeted actions",
            Some(actions_json),
        )?
        .into_iter()
        .map(TryInto::try_into)
        .collect::<PyResult<Vec<_>>>()?;
        to_json(&self.inner.step_targeted(actions))
    }

    fn teacher_action_json(&mut self, role: String) -> PyResult<String> {
        let role = parse_aircraft_role(&role)?;
        to_json(&self.inner.teacher_action(role).map_err(py_runtime_error)?)
    }

    fn set_control_authority(&mut self, role: String, authority: String) -> PyResult<()> {
        self.inner.set_control_authority(
            parse_aircraft_role(&role)?,
            parse_control_authority(&authority)?,
        );
        Ok(())
    }

    #[pyo3(signature = (
        enable_visual=false,
        enable_audio=false,
        visual_sensors_json=None,
        audio_window_seconds=0.25
    ))]
    fn start_recording(
        &mut self,
        enable_visual: bool,
        enable_audio: bool,
        visual_sensors_json: Option<String>,
        audio_window_seconds: f32,
    ) -> PyResult<()> {
        self.inner
            .start_recording(Some(build_capture_config(
                enable_visual,
                enable_audio,
                visual_sensors_json,
                audio_window_seconds,
            )?))
            .map_err(py_runtime_error)
    }

    fn stop_recording(&mut self) -> bool {
        self.inner.stop_recording()
    }

    fn shutdown(&mut self) {
        self.inner.shutdown();
    }
}

#[pymethods]
impl PyEpisodeRecording {
    #[new]
    fn new(episode_root: String) -> Self {
        Self {
            inner: RecordingAccess::new(episode_root),
        }
    }

    fn episode_root(&self) -> String {
        self.inner.episode_root().display().to_string()
    }

    fn manifest_json(&self) -> PyResult<String> {
        to_json(&self.inner.manifest().map_err(py_runtime_error)?)
    }

    fn load_episode_json(&self) -> PyResult<String> {
        to_json(&self.inner.load_episode().map_err(py_runtime_error)?)
    }

    fn initial_snapshot_json(&self) -> PyResult<String> {
        to_json(&self.inner.initial_snapshot().map_err(py_runtime_error)?)
    }

    fn step_json(&self, index: u32) -> PyResult<String> {
        to_json(&self.inner.step(index).map_err(py_runtime_error)?)
    }

    fn steps_json(&self) -> PyResult<String> {
        to_json(&self.inner.steps().map_err(py_runtime_error)?)
    }

    fn step_artifacts_json(&self, index: u32) -> PyResult<String> {
        to_json(
            &self
                .inner
                .step_artifacts_at(index)
                .map_err(py_runtime_error)?,
        )
    }

    fn all_step_artifacts_json(&self) -> PyResult<String> {
        to_json(&self.inner.step_artifacts().map_err(py_runtime_error)?)
    }

    fn derived_root_for_role(&self, role: String) -> String {
        self.inner
            .derived_root_for_role(&role)
            .display()
            .to_string()
    }

    fn validation_root_for_role(&self, role: String) -> PyResult<String> {
        Ok(self
            .inner
            .validation_root_for_role(&role)
            .map_err(py_runtime_error)?
            .display()
            .to_string())
    }

    fn available_derived_roles_json(&self) -> PyResult<String> {
        to_json(
            &self
                .inner
                .available_derived_roles()
                .map_err(py_runtime_error)?,
        )
    }

    fn derived_manifest_json(&self, role: String) -> PyResult<String> {
        to_json(
            &self
                .inner
                .derived_manifest(&role)
                .map_err(py_runtime_error)?,
        )
    }

    fn validation_audio_path(&self, role: String) -> PyResult<String> {
        Ok(self
            .inner
            .validation_audio_path(&role)
            .map_err(py_runtime_error)?
            .display()
            .to_string())
    }

    fn validation_video_path(&self, role: String, camera: String) -> PyResult<String> {
        Ok(self
            .inner
            .validation_video_path(&role, &camera)
            .map_err(py_runtime_error)?
            .display()
            .to_string())
    }

    fn step_visual_path(&self, index: u32, camera: String) -> PyResult<Option<String>> {
        let artifacts = self
            .inner
            .step_artifacts_at(index)
            .map_err(py_runtime_error)?;
        Ok(artifacts
            .visual
            .into_iter()
            .find(|artifact| visual_kind_matches(artifact.camera, &camera))
            .and_then(|artifact| artifact.file_path))
    }

    fn step_audio_path(&self, index: u32) -> PyResult<Option<String>> {
        Ok(self
            .inner
            .step_artifacts_at(index)
            .map_err(py_runtime_error)?
            .audio
            .and_then(|artifact| artifact.file_path))
    }

    fn step_visual_bytes<'py>(
        &self,
        py: Python<'py>,
        index: u32,
        camera: String,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let camera = parse_visual_sensor_kind(&camera)?;
        let Some(artifact) = self
            .inner
            .step_artifacts_at(index)
            .map_err(py_runtime_error)?
            .visual
            .into_iter()
            .find(|artifact| artifact.camera == camera)
        else {
            return Ok(None);
        };
        let bytes = self
            .inner
            .read_visual_artifact_bytes(&artifact)
            .map_err(py_runtime_error)?;
        Ok(Some(PyBytes::new(py, &bytes)))
    }

    fn step_audio_bytes<'py>(
        &self,
        py: Python<'py>,
        index: u32,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let Some(artifact) = self
            .inner
            .step_artifacts_at(index)
            .map_err(py_runtime_error)?
            .audio
        else {
            return Ok(None);
        };
        let bytes = self
            .inner
            .read_audio_artifact_bytes(&artifact)
            .map_err(py_runtime_error)?;
        Ok(Some(PyBytes::new(py, &bytes)))
    }

    fn read_bytes<'py>(
        &self,
        py: Python<'py>,
        relative_path: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .inner
            .read_relative_bytes(relative_path)
            .map_err(py_runtime_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

#[pymethods]
impl PyEpisodeReconstructor {
    #[new]
    #[pyo3(signature = (
        episode_root,
        observed_role="fighter1".to_string(),
        enable_visual=false,
        enable_audio=false,
        visual_sensors_json=None,
        audio_window_seconds=0.25
    ))]
    fn new(
        episode_root: String,
        observed_role: String,
        enable_visual: bool,
        enable_audio: bool,
        visual_sensors_json: Option<String>,
        audio_window_seconds: f32,
    ) -> PyResult<Self> {
        let capture_config = build_capture_config(
            enable_visual,
            enable_audio,
            visual_sensors_json,
            audio_window_seconds,
        )?;
        let observed_role = parse_aircraft_role(&observed_role)?;
        Ok(Self {
            inner: RecordingReconstructionSession::new(episode_root, observed_role, capture_config)
                .map_err(py_runtime_error)?,
        })
    }

    fn manifest_json(&self) -> PyResult<String> {
        to_json(self.inner.manifest())
    }

    fn reconstruct_initial_json(&mut self) -> PyResult<String> {
        to_json(
            &self
                .inner
                .reconstruct_initial_observation()
                .map_err(py_runtime_error)?,
        )
    }

    fn reconstruct_initial_visual_bytes<'py>(
        &mut self,
        py: Python<'py>,
        camera: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let observation = self
            .inner
            .reconstruct_initial_observation()
            .map_err(py_runtime_error)?;
        let frame = find_visual_frame(&observation, &camera)?;
        Ok(PyBytes::new(py, &frame.bytes))
    }

    fn reconstruct_initial_audio_samples_bytes<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let observation = self
            .inner
            .reconstruct_initial_observation()
            .map_err(py_runtime_error)?;
        let audio = find_audio_observation(&observation)?;
        Ok(PyBytes::new(py, &encode_f32_samples_le(&audio.samples)))
    }

    fn reconstruct_step_json(&mut self, index: u32) -> PyResult<String> {
        to_json(
            &self
                .inner
                .reconstruct_step_observation(index)
                .map_err(py_runtime_error)?,
        )
    }

    fn reconstruct_step_visual_bytes<'py>(
        &mut self,
        py: Python<'py>,
        index: u32,
        camera: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let observation = self
            .inner
            .reconstruct_step_observation(index)
            .map_err(py_runtime_error)?;
        let frame = find_visual_frame(&observation, &camera)?;
        Ok(PyBytes::new(py, &frame.bytes))
    }

    fn reconstruct_step_audio_samples_bytes<'py>(
        &mut self,
        py: Python<'py>,
        index: u32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let observation = self
            .inner
            .reconstruct_step_observation(index)
            .map_err(py_runtime_error)?;
        let audio = find_audio_observation(&observation)?;
        Ok(PyBytes::new(py, &encode_f32_samples_le(&audio.samples)))
    }
}

#[pymodule]
pub fn dfb_game(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyEnvironment>()?;
    module.add_class::<PyEnvironmentAction>()?;
    module.add_class::<PyEpisodeRecording>()?;
    module.add_class::<PyEpisodeReconstructor>()?;
    Ok(())
}
