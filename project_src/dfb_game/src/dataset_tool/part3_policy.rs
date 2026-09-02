use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail, ensure};
use bevy::math::{Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::api::types::{AircraftObservation, StateObservation};
use crate::policy_contract::CONTRACT_BYTES;
pub use crate::policy_contract::{
    ACTION_BIN_DIM, ACTION_CONT_DIM, ACTION_SCHEMA_ID, BINARY_OBS_INDICES, CHECKPOINT_SCHEMA_ID,
    DATASET_SCHEMA_ID, MODEL_FAMILY_ID, NORMALIZER_SCHEMA_ID, OBS_DIM, OBSERVATION_SCHEMA_ID,
    POLICY_CONTRACT_ID, policy_contract_sha256,
};
const HEALTH_SUBSYSTEMS: [&str; 5] = ["LeftWing", "RightWing", "PitchTail", "YawTail", "Engine"];

#[derive(Debug, Deserialize)]
struct PolicyContract {
    policy_contract_id: String,
    normalizer_schema_id: String,
    dataset_schema_id: String,
    checkpoint_schema_id: String,
    model_family_id: String,
    observation: ObservationContract,
    action: ActionContract,
    geometry: GeometryContract,
}

#[derive(Debug, Deserialize)]
struct ObservationContract {
    schema_id: String,
    dim: usize,
    fields: Vec<ObservationFieldContract>,
    binary_indices: Vec<usize>,
    scales: ObservationScales,
}

#[derive(Debug, Deserialize)]
struct ObservationFieldContract {
    name: String,
    offset: usize,
    size: usize,
}

#[derive(Debug, Deserialize)]
struct ObservationScales {
    relative_position_m: f32,
    world_position_m: f32,
    linear_velocity_mps: f32,
    angular_velocity_rad_s: f32,
    total_hit_points: f32,
    gun_heat: f32,
    repair_seconds: f32,
    out_of_bounds_seconds: f32,
    episode_seconds: f32,
}

#[derive(Debug, Deserialize)]
struct ActionContract {
    schema_id: String,
    continuous: Vec<String>,
    binary: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GeometryContract {
    projectile_speed_mps: f32,
    projectile_max_range_m: f32,
    muzzle_forward_offset_m: f32,
    attack_tau_reference_seconds: f32,
    fire_alignment_threshold_cos: f32,
    projectile_aircraft_hit_radius_m: f32,
    projectile_subsystem_hit_radius_m: f32,
    shot_outer_radius_m: f32,
    shot_core_radius_m: f32,
    shot_outer_weight: f32,
    shot_core_weight: f32,
    collision_boxes: Vec<CollisionBoxContract>,
}

#[derive(Debug, Deserialize)]
struct CollisionBoxContract {
    name: String,
    center: [f32; 3],
    half_extents: [f32; 3],
    subsystem: bool,
}

#[derive(Debug, Clone, Copy)]
struct WorldCollisionBox {
    center: Vec3,
    rotation: Mat3,
    half_extents: Vec3,
    subsystem: bool,
}

#[derive(Debug, Clone, Copy)]
struct AttackGeometry {
    tracking_quality: f32,
    tail_hold_score: f32,
    shot_feasibility: f32,
}

#[derive(Debug, Deserialize)]
struct FixtureSet {
    cases: Vec<FixtureCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    name: String,
    episode_start_sim_time_seconds: f32,
    state: StateObservation,
}

#[derive(Debug, Serialize)]
struct FixtureOutput {
    policy_contract_id: &'static str,
    observation_schema_id: &'static str,
    contract_sha256: String,
    cases: Vec<FixtureCaseOutput>,
}

#[derive(Debug, Serialize)]
struct FixtureCaseOutput {
    name: String,
    roles: BTreeMap<String, Vec<f32>>,
}

fn contract() -> &'static PolicyContract {
    static CONTRACT: OnceLock<PolicyContract> = OnceLock::new();
    CONTRACT.get_or_init(|| {
        let parsed: PolicyContract = serde_json::from_slice(CONTRACT_BYTES)
            .expect("embedded Part 3 policy contract must be valid JSON");
        validate_contract(&parsed).expect("embedded Part 3 policy contract must be valid");
        parsed
    })
}

fn validate_contract(contract: &PolicyContract) -> Result<()> {
    ensure!(contract.policy_contract_id == POLICY_CONTRACT_ID);
    ensure!(contract.observation.schema_id == OBSERVATION_SCHEMA_ID);
    ensure!(contract.normalizer_schema_id == NORMALIZER_SCHEMA_ID);
    ensure!(contract.dataset_schema_id == DATASET_SCHEMA_ID);
    ensure!(contract.checkpoint_schema_id == CHECKPOINT_SCHEMA_ID);
    ensure!(contract.model_family_id == MODEL_FAMILY_ID);
    ensure!(contract.action.schema_id == ACTION_SCHEMA_ID);
    ensure!(contract.observation.dim == OBS_DIM);
    ensure!(contract.observation.binary_indices == BINARY_OBS_INDICES);
    ensure!(contract.action.continuous == ["throttle_delta", "pitch", "roll", "yaw"]);
    ensure!(contract.action.binary == ["brake", "fire_gun", "repair"]);
    let mut expected_offset = 0;
    let mut names = BTreeSet::new();
    for field in &contract.observation.fields {
        ensure!(
            field.offset == expected_offset,
            "non-contiguous field {}",
            field.name
        );
        ensure!(field.size > 0, "empty field {}", field.name);
        ensure!(names.insert(&field.name), "duplicate field {}", field.name);
        expected_offset += field.size;
    }
    ensure!(expected_offset == OBS_DIM);
    Ok(())
}

pub fn run_fixture_from_args<I>(args: I) -> Result<()>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let fixture_path = match (args.next(), args.next()) {
        (Some(flag), Some(path)) if flag == "--fixture" => PathBuf::from(path),
        _ => bail!("usage: part3-policy-observation-fixture --fixture <path>"),
    };
    ensure!(args.next().is_none(), "unexpected fixture arguments");
    let output = build_fixture_output(&fixture_path)?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

fn build_fixture_output(path: &Path) -> Result<FixtureOutput> {
    let fixture: FixtureSet = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", path.display()))?;
    let mut cases = Vec::with_capacity(fixture.cases.len());
    for case in fixture.cases {
        let mut roles = BTreeMap::new();
        for role in ["fighter1", "fighter2"] {
            roles.insert(
                role.to_string(),
                build_policy_observation(&case.state, role, case.episode_start_sim_time_seconds)?
                    .to_vec(),
            );
        }
        cases.push(FixtureCaseOutput {
            name: case.name,
            roles,
        });
    }
    Ok(FixtureOutput {
        policy_contract_id: POLICY_CONTRACT_ID,
        observation_schema_id: OBSERVATION_SCHEMA_ID,
        contract_sha256: policy_contract_sha256(),
        cases,
    })
}

pub fn build_policy_observation(
    state: &StateObservation,
    role: &str,
    episode_start_sim_time_seconds: f32,
) -> Result<[f32; OBS_DIM]> {
    ensure!(
        state.aircraft.len() == 2,
        "state must contain exactly two aircraft"
    );
    ensure!(
        state.sim_time_seconds.is_finite(),
        "sim time must be finite"
    );
    ensure!(
        episode_start_sim_time_seconds.is_finite(),
        "episode start sim time must be finite"
    );
    ensure!(
        state.arena.arena_radius.is_finite() && state.arena.arena_radius > 0.0,
        "arena radius must be finite and positive"
    );
    let ego = find_unique_aircraft(state, role)?;
    let enemy = state
        .aircraft
        .iter()
        .find(|aircraft| aircraft.role != role)
        .ok_or_else(|| anyhow!("missing enemy for role {role}"))?;
    validate_aircraft(ego)?;
    validate_aircraft(enemy)?;

    let config = contract();
    let scales = &config.observation.scales;
    let ego_position = finite_vec3(ego.position, "ego.position")?;
    let enemy_position = finite_vec3(enemy.position, "enemy.position")?;
    let ego_velocity = finite_vec3(ego.linear_velocity, "ego.linear_velocity")?;
    let enemy_velocity = finite_vec3(enemy.linear_velocity, "enemy.linear_velocity")?;
    let ego_angular_velocity = finite_vec3(ego.angular_velocity_deg, "ego.angular_velocity_deg")?;
    let enemy_angular_velocity =
        finite_vec3(enemy.angular_velocity_deg, "enemy.angular_velocity_deg")?;
    let ego_rotation = rotation_from_quaternion(ego.orientation_quat, "ego.orientation_quat")?;
    let enemy_rotation =
        rotation_from_quaternion(enemy.orientation_quat, "enemy.orientation_quat")?;
    let self_attack = compute_attack_geometry(ego, enemy)?;
    let enemy_attack = compute_attack_geometry(enemy, ego)?;
    let self_health = health_state(ego, scales.total_hit_points)?;
    let enemy_health = health_state(enemy, scales.total_hit_points)?;

    let enemy_oob_active = enemy.out_of_bounds_seconds > 0.0
        || Vec3::new(enemy_position.x, 0.0, enemy_position.z).length() >= state.arena.arena_radius;
    let self_oob_active = ego.out_of_bounds_seconds > 0.0
        || Vec3::new(ego_position.x, 0.0, ego_position.z).length() >= state.arena.arena_radius;

    let mut observation = [0.0; OBS_DIM];
    let mut cursor = 0;
    push_vec3(
        &mut observation,
        &mut cursor,
        ego_rotation.transpose() * (enemy_position - ego_position) / scales.relative_position_m,
    );
    push_slice(
        &mut observation,
        &mut cursor,
        &rotation_6d(ego_rotation.transpose() * enemy_rotation),
    );
    push_vec3(
        &mut observation,
        &mut cursor,
        enemy_rotation.transpose() * enemy_velocity / scales.linear_velocity_mps,
    );
    push_vec3(
        &mut observation,
        &mut cursor,
        degrees_to_radians(enemy_angular_velocity) / scales.angular_velocity_rad_s,
    );
    push_slice(&mut observation, &mut cursor, &enemy_health);
    push_scalar(
        &mut observation,
        &mut cursor,
        enemy.throttle.clamp(0.0, 1.0),
    );
    push_scalar(&mut observation, &mut cursor, f32::from(enemy.brake));
    push_scalar(
        &mut observation,
        &mut cursor,
        enemy.stall_factor.clamp(0.0, 1.0),
    );
    push_scalar(
        &mut observation,
        &mut cursor,
        f32::from(enemy.gun_overheated),
    );
    push_scalar(
        &mut observation,
        &mut cursor,
        (enemy.gun_heat / scales.gun_heat).clamp(0.0, 4.0),
    );
    push_scalar(&mut observation, &mut cursor, f32::from(enemy.is_firing));
    push_scalar(&mut observation, &mut cursor, f32::from(enemy.repairing));
    push_scalar(
        &mut observation,
        &mut cursor,
        normalized_duration(
            enemy.repairing,
            enemy.repair_elapsed_seconds,
            scales.repair_seconds,
        ),
    );
    push_scalar(&mut observation, &mut cursor, f32::from(enemy_oob_active));
    push_scalar(
        &mut observation,
        &mut cursor,
        normalized_duration(
            enemy_oob_active,
            enemy.out_of_bounds_seconds,
            scales.out_of_bounds_seconds,
        ),
    );
    push_scalar(&mut observation, &mut cursor, enemy_attack.tracking_quality);
    push_scalar(&mut observation, &mut cursor, enemy_attack.tail_hold_score);
    push_scalar(&mut observation, &mut cursor, enemy_attack.shot_feasibility);
    push_scalar(
        &mut observation,
        &mut cursor,
        ((state.sim_time_seconds - episode_start_sim_time_seconds).max(0.0)
            / scales.episode_seconds)
            .clamp(0.0, 4.0),
    );
    push_vec3(
        &mut observation,
        &mut cursor,
        ego_position / scales.world_position_m,
    );
    push_slice(&mut observation, &mut cursor, &rotation_6d(ego_rotation));
    push_scalar(&mut observation, &mut cursor, ego.throttle.clamp(0.0, 1.0));
    push_scalar(&mut observation, &mut cursor, f32::from(ego.brake));
    push_scalar(
        &mut observation,
        &mut cursor,
        ego.stall_factor.clamp(0.0, 1.0),
    );
    push_vec3(
        &mut observation,
        &mut cursor,
        ego_rotation.transpose() * ego_velocity / scales.linear_velocity_mps,
    );
    push_vec3(
        &mut observation,
        &mut cursor,
        degrees_to_radians(ego_angular_velocity) / scales.angular_velocity_rad_s,
    );
    push_slice(&mut observation, &mut cursor, &self_health);
    push_scalar(&mut observation, &mut cursor, f32::from(ego.gun_overheated));
    push_scalar(
        &mut observation,
        &mut cursor,
        (ego.gun_heat / scales.gun_heat).clamp(0.0, 4.0),
    );
    push_scalar(&mut observation, &mut cursor, f32::from(ego.is_firing));
    push_scalar(&mut observation, &mut cursor, f32::from(ego.repairing));
    push_scalar(
        &mut observation,
        &mut cursor,
        normalized_duration(
            ego.repairing,
            ego.repair_elapsed_seconds,
            scales.repair_seconds,
        ),
    );
    push_scalar(&mut observation, &mut cursor, f32::from(self_oob_active));
    push_scalar(
        &mut observation,
        &mut cursor,
        normalized_duration(
            self_oob_active,
            ego.out_of_bounds_seconds,
            scales.out_of_bounds_seconds,
        ),
    );
    push_scalar(&mut observation, &mut cursor, self_attack.tracking_quality);
    push_scalar(&mut observation, &mut cursor, self_attack.tail_hold_score);
    push_scalar(&mut observation, &mut cursor, self_attack.shot_feasibility);
    ensure!(cursor == OBS_DIM, "observation cursor mismatch");
    ensure!(
        observation.iter().all(|value| value.is_finite()),
        "observation contains non-finite values"
    );
    Ok(observation)
}

fn find_unique_aircraft<'a>(
    state: &'a StateObservation,
    role: &str,
) -> Result<&'a AircraftObservation> {
    let matches = state
        .aircraft
        .iter()
        .filter(|aircraft| aircraft.role == role)
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "expected exactly one aircraft role {role}"
    );
    Ok(matches[0])
}

fn validate_aircraft(aircraft: &AircraftObservation) -> Result<()> {
    ensure!(!aircraft.role.is_empty(), "aircraft role must not be empty");
    finite_vec3(aircraft.position, "position")?;
    finite_vec3(aircraft.linear_velocity, "linear_velocity")?;
    finite_vec3(aircraft.angular_velocity_deg, "angular_velocity_deg")?;
    rotation_from_quaternion(aircraft.orientation_quat, "orientation_quat")?;
    for (name, value) in [
        ("throttle", aircraft.throttle),
        ("stall_factor", aircraft.stall_factor),
        ("hit_points", aircraft.hit_points),
        ("gun_heat", aircraft.gun_heat),
        ("repair_elapsed_seconds", aircraft.repair_elapsed_seconds),
        ("out_of_bounds_seconds", aircraft.out_of_bounds_seconds),
    ] {
        ensure!(value.is_finite(), "{name} must be finite");
    }
    health_state(aircraft, contract().observation.scales.total_hit_points)?;
    Ok(())
}

fn health_state(aircraft: &AircraftObservation, total_max: f32) -> Result<[f32; 6]> {
    ensure!(total_max.is_finite() && total_max > 0.0);
    let mut by_name = BTreeMap::new();
    for subsystem in &aircraft.subsystems {
        ensure!(
            by_name.insert(subsystem.name.as_str(), subsystem).is_none(),
            "duplicate subsystem {}",
            subsystem.name
        );
    }
    let mut output = [0.0; 6];
    output[0] = (aircraft.hit_points / total_max).clamp(0.0, 1.0);
    for (offset, name) in HEALTH_SUBSYSTEMS.iter().enumerate() {
        let subsystem = by_name
            .get(name)
            .ok_or_else(|| anyhow!("missing subsystem {name}"))?;
        ensure!(
            subsystem.hit_points.is_finite()
                && subsystem.max_hit_points.is_finite()
                && subsystem.max_hit_points > 0.0,
            "invalid subsystem health for {name}"
        );
        output[offset + 1] = (subsystem.hit_points / subsystem.max_hit_points).clamp(0.0, 1.0);
    }
    Ok(output)
}

fn compute_attack_geometry(
    attacker: &AircraftObservation,
    defender: &AircraftObservation,
) -> Result<AttackGeometry> {
    let geometry = &contract().geometry;
    let attacker_position = finite_vec3(attacker.position, "attacker.position")?;
    let defender_position = finite_vec3(defender.position, "defender.position")?;
    let attacker_velocity = finite_vec3(attacker.linear_velocity, "attacker.linear_velocity")?;
    let attacker_rotation =
        rotation_from_quaternion(attacker.orientation_quat, "attacker.orientation_quat")?;
    let defender_rotation =
        rotation_from_quaternion(defender.orientation_quat, "defender.orientation_quat")?;
    let attacker_forward = attacker_rotation.z_axis;
    let defender_forward = defender_rotation.z_axis;
    let relative_position = defender_position - attacker_position;
    let distance = relative_position.length();
    ensure!(
        distance > 1e-6,
        "attacker and defender positions must differ"
    );
    let line_of_sight = relative_position / distance;
    let aim_cos = attacker_forward.dot(line_of_sight).clamp(-1.0, 1.0);
    let tail_cos = defender_forward.dot(line_of_sight).clamp(-1.0, 1.0);
    let heading_cos = attacker_forward.dot(defender_forward).clamp(-1.0, 1.0);
    let tracking_quality = (0.5 * (aim_cos + 1.0)).clamp(0.0, 1.0);
    let tail_hold_score =
        (0.5 * (tail_cos + 1.0)).clamp(0.0, 1.0) * (0.5 * (heading_cos + 1.0)).clamp(0.0, 1.0);

    let muzzle = attacker_position + attacker_forward * geometry.muzzle_forward_offset_m;
    let bullet_velocity = attacker_velocity + attacker_forward * geometry.projectile_speed_mps;
    let (outer_score, core_score) = projectile_box_hit_scores(muzzle, bullet_velocity, defender)?;
    let fire_alignment = centered_cone_score(aim_cos, geometry.fire_alignment_threshold_cos);
    let shot_feasibility = (fire_alignment
        * (geometry.shot_outer_weight * outer_score + geometry.shot_core_weight * core_score))
        .clamp(0.0, 1.0);
    Ok(AttackGeometry {
        tracking_quality,
        tail_hold_score,
        shot_feasibility,
    })
}

fn projectile_box_hit_scores(
    muzzle: Vec3,
    bullet_velocity: Vec3,
    defender: &AircraftObservation,
) -> Result<(f32, f32)> {
    let geometry = &contract().geometry;
    let bullet_speed = bullet_velocity.length();
    if bullet_speed <= 1e-6 {
        return Ok((0.0, 0.0));
    }
    let defender_velocity = finite_vec3(defender.linear_velocity, "defender.linear_velocity")?;
    let relative_velocity = defender_velocity - bullet_velocity;
    let tau_max = geometry.projectile_max_range_m / bullet_speed;
    let mut outer_best: f32 = 0.0;
    let mut core_best: f32 = 0.0;
    for collision_box in world_collision_boxes(defender)? {
        outer_best = outer_best.max(box_hit_score(
            collision_box,
            muzzle,
            relative_velocity,
            tau_max,
            geometry.projectile_aircraft_hit_radius_m,
            geometry.shot_outer_radius_m,
        ));
        if collision_box.subsystem {
            core_best = core_best.max(box_hit_score(
                collision_box,
                muzzle,
                relative_velocity,
                tau_max,
                geometry.projectile_subsystem_hit_radius_m,
                geometry.shot_core_radius_m,
            ));
        }
    }
    Ok((outer_best.clamp(0.0, 1.0), core_best.clamp(0.0, 1.0)))
}

fn world_collision_boxes(defender: &AircraftObservation) -> Result<Vec<WorldCollisionBox>> {
    let position = finite_vec3(defender.position, "defender.position")?;
    let rotation =
        rotation_from_quaternion(defender.orientation_quat, "defender.orientation_quat")?;
    let destroyed = defender
        .subsystems
        .iter()
        .filter(|subsystem| subsystem.stage == "Destroyed")
        .map(|subsystem| subsystem.name.as_str())
        .collect::<BTreeSet<_>>();
    Ok(contract()
        .geometry
        .collision_boxes
        .iter()
        .filter(|collision_box| !destroyed.contains(collision_box.name.as_str()))
        .map(|collision_box| WorldCollisionBox {
            center: position + rotation * Vec3::from_array(collision_box.center),
            rotation,
            half_extents: Vec3::from_array(collision_box.half_extents),
            subsystem: collision_box.subsystem,
        })
        .collect())
}

fn box_hit_score(
    collision_box: WorldCollisionBox,
    muzzle: Vec3,
    relative_velocity: Vec3,
    tau_max: f32,
    hit_radius: f32,
    score_radius: f32,
) -> f32 {
    let geometry = &contract().geometry;
    let relative_position = collision_box.center - muzzle;
    let (tau, center_distance, closest_delta) =
        closest_approach(relative_position, relative_velocity, tau_max);
    let closest_direction = if closest_delta.length() <= 1e-6 {
        Vec3::X
    } else {
        closest_delta.normalize()
    };
    let support = box_support_radius(collision_box, -closest_direction);
    let clearance = (center_distance - support - hit_radius).max(0.0);
    let tau_gate = (-tau / geometry.attack_tau_reference_seconds).exp();
    tau_gate * (-(clearance / score_radius.max(hit_radius)).powi(2)).exp()
}

fn closest_approach(
    relative_position: Vec3,
    relative_velocity: Vec3,
    horizon_seconds: f32,
) -> (f32, f32, Vec3) {
    let speed_squared = relative_velocity.length_squared();
    let tau = if speed_squared <= 1e-6 {
        0.0
    } else {
        (-relative_position.dot(relative_velocity) / speed_squared).clamp(0.0, horizon_seconds)
    };
    let closest_delta = relative_position + relative_velocity * tau;
    (tau, closest_delta.length(), closest_delta)
}

fn box_support_radius(collision_box: WorldCollisionBox, direction: Vec3) -> f32 {
    if direction.length() <= 1e-6 {
        return collision_box.half_extents.max_element();
    }
    let direction = direction.normalize();
    collision_box.half_extents.x * collision_box.rotation.x_axis.dot(direction).abs()
        + collision_box.half_extents.y * collision_box.rotation.y_axis.dot(direction).abs()
        + collision_box.half_extents.z * collision_box.rotation.z_axis.dot(direction).abs()
}

fn centered_cone_score(cos_value: f32, threshold: f32) -> f32 {
    if cos_value <= threshold {
        0.0
    } else {
        ((cos_value - threshold) / (1.0 - threshold).max(1e-6)).powi(2)
    }
}

fn finite_vec3(value: [f32; 3], name: &str) -> Result<Vec3> {
    let vector = Vec3::from_array(value);
    ensure!(vector.is_finite(), "{name} must be finite");
    Ok(vector)
}

fn rotation_from_quaternion(value: [f32; 4], name: &str) -> Result<Mat3> {
    let quaternion = Quat::from_array(value);
    ensure!(quaternion.is_finite(), "{name} must be finite");
    ensure!(
        quaternion.length_squared() > 1e-12,
        "{name} must have non-zero length"
    );
    Ok(Mat3::from_quat(quaternion.normalize()))
}

fn rotation_6d(rotation: Mat3) -> [f32; 6] {
    let x = rotation.x_axis;
    let y = rotation.y_axis;
    [x.x, x.y, x.z, y.x, y.y, y.z]
}

fn degrees_to_radians(value: Vec3) -> Vec3 {
    Vec3::new(
        value.x.to_radians(),
        value.y.to_radians(),
        value.z.to_radians(),
    )
}

fn normalized_duration(active: bool, elapsed_seconds: f32, scale_seconds: f32) -> f32 {
    if active {
        (elapsed_seconds / scale_seconds).clamp(0.0, 4.0)
    } else {
        0.0
    }
}

fn push_scalar(buffer: &mut [f32; OBS_DIM], cursor: &mut usize, value: f32) {
    buffer[*cursor] = value;
    *cursor += 1;
}

fn push_vec3(buffer: &mut [f32; OBS_DIM], cursor: &mut usize, value: Vec3) {
    push_slice(buffer, cursor, &value.to_array());
}

fn push_slice<const N: usize>(buffer: &mut [f32; OBS_DIM], cursor: &mut usize, values: &[f32; N]) {
    buffer[*cursor..*cursor + N].copy_from_slice(values);
    *cursor += N;
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn embedded_contract_has_expected_identity_and_layout() {
        let contract = contract();
        assert_eq!(contract.policy_contract_id, POLICY_CONTRACT_ID);
        assert_eq!(contract.observation.dim, OBS_DIM);
        assert_eq!(contract.observation.binary_indices, BINARY_OBS_INDICES);
        assert_eq!(policy_contract_sha256().len(), 64);
    }

    #[test]
    fn rotation_6d_is_column_major() {
        let matrix = Mat3::from_cols(
            Vec3::new(1.0, 4.0, 7.0),
            Vec3::new(2.0, 5.0, 8.0),
            Vec3::new(3.0, 6.0, 9.0),
        );
        assert_eq!(rotation_6d(matrix), [1.0, 4.0, 7.0, 2.0, 5.0, 8.0]);
    }

    #[test]
    fn zero_quaternion_is_rejected() {
        assert!(rotation_from_quaternion([0.0; 4], "orientation").is_err());
    }

    #[test]
    fn normalized_duration_requires_explicit_activation() {
        assert_eq!(normalized_duration(false, 12.0, 10.0), 0.0);
        assert_eq!(normalized_duration(true, 0.0, 10.0), 0.0);
        assert_eq!(normalized_duration(true, 12.0, 10.0), 1.2);
    }

    #[test]
    fn shared_fixture_covers_required_combat_geometries() {
        let fixture: FixtureSet = serde_json::from_slice(include_bytes!(
            "../../../../config/dfb_reinforcement_learning/fixtures/part3_policy_observation_v1_cases.json"
        ))
        .expect("shared fixture must parse");
        let names = fixture
            .cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            names,
            BTreeSet::from(["crossing", "head_on", "rotation", "tail_chase"])
        );
        for case in fixture.cases {
            for role in ["fighter1", "fighter2"] {
                let observation = build_policy_observation(
                    &case.state,
                    role,
                    case.episode_start_sim_time_seconds,
                )
                .expect("fixture observation must build");
                assert!(observation.iter().all(|value| value.is_finite()));
            }
        }
    }
}
