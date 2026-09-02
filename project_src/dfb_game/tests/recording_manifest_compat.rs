use dfb_game::api::types::ObservationCaptureConfig;
use dfb_game::policy_contract::{ACTION_SCHEMA_ID, POLICY_CONTRACT_ID};
use dfb_game::recording::{
    RecordedEpisodeManifest, RecordingArtifactConvention, UNSPECIFIED_POLICY_CONTRACT_ID,
};

fn manifest_fixture() -> RecordedEpisodeManifest {
    RecordedEpisodeManifest {
        schema_version: 12,
        policy_contract_id: POLICY_CONTRACT_ID.to_string(),
        action_schema_id: ACTION_SCHEMA_ID.to_string(),
        episode_id: "compatibility_fixture".to_string(),
        scene_name: "open_head_on".to_string(),
        seed: Some(42),
        fixed_time_step_seconds: 1.0 / 60.0,
        capture_config: ObservationCaptureConfig::default(),
        audio_artifact_metadata: None,
        started_tick: 0,
        started_sim_time_seconds: 0.0,
        final_tick: 0,
        final_sim_time_seconds: 0.0,
        total_steps: 0,
        step_chunks: Vec::new(),
        step_artifacts: Vec::new(),
        termination_reason: "manual_stop".to_string(),
        winner: None,
        bridge_role: None,
        bridge_session: None,
        bridge_transport: None,
        authoritative_source: true,
        initial_state_path: "initial_state.ron".to_string(),
        artifact_convention: RecordingArtifactConvention::default(),
    }
}

#[test]
fn legacy_manifest_without_policy_contract_remains_replayable() {
    let encoded = ron::to_string(&manifest_fixture()).expect("manifest should serialize");
    let declared_contract = format!("policy_contract_id:\"{POLICY_CONTRACT_ID}\",");
    assert!(encoded.contains(&declared_contract));

    let legacy_encoded = encoded.replacen(&declared_contract, "", 1);
    let decoded: RecordedEpisodeManifest =
        ron::from_str(&legacy_encoded).expect("legacy manifest should deserialize");

    assert_eq!(decoded.policy_contract_id, UNSPECIFIED_POLICY_CONTRACT_ID);
    assert_ne!(decoded.policy_contract_id, POLICY_CONTRACT_ID);
}

#[test]
fn current_manifest_keeps_explicit_policy_contract() {
    let encoded = ron::to_string(&manifest_fixture()).expect("manifest should serialize");
    let decoded: RecordedEpisodeManifest =
        ron::from_str(&encoded).expect("current manifest should deserialize");

    assert_eq!(decoded.policy_contract_id, POLICY_CONTRACT_ID);
}
