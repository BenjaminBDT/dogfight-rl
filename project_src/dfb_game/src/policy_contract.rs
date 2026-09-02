use sha2::{Digest, Sha256};

pub const POLICY_CONTRACT_ID: &str = "dfb_part3_policy_contract_v1";
pub const OBSERVATION_SCHEMA_ID: &str = "dfb_part3_policy_observation_v1";
pub const ACTION_SCHEMA_ID: &str = "dfb_part3_policy_action_v1";
pub const NORMALIZER_SCHEMA_ID: &str = "dfb_part3_policy_normalizer_v1";
pub const DATASET_SCHEMA_ID: &str = "dfb_part3_policy_dataset_v1";
pub const CHECKPOINT_SCHEMA_ID: &str = "dfb_part3_policy_checkpoint_v1";
pub const MODEL_FAMILY_ID: &str = "dfb_part3_stateless_hybrid_actor_critic_v1";
pub const OBS_DIM: usize = 69;
pub const ACTION_CONT_DIM: usize = 4;
pub const ACTION_BIN_DIM: usize = 3;
pub const BINARY_OBS_INDICES: [usize; 10] = [22, 24, 26, 27, 29, 45, 59, 61, 62, 64];

pub const CONTRACT_BYTES: &[u8] =
    include_bytes!("../../../config/dfb_reinforcement_learning/part3_policy_contract_v1.json");

pub fn policy_contract_sha256() -> String {
    hex::encode(Sha256::digest(CONTRACT_BYTES))
}
