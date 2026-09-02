use dfb_game::app::game_app::build_client_app_with_paths;
use dfb_game::bridge::ipc_stub::{BridgeEndpointConfig, BridgeTransport};
use dfb_game::bridge::protocol::{BridgeControlSlot, LobbyClientKind};
use dfb_game::bridge::{BridgeLaunchBinding, LocalPilotMode, RequestedControlRole};
use dfb_game::core::config::{ConfigPaths, resolve_project_root};
use dfb_game::model_control::ModelControlConfig;
use std::env;

fn main() {
    let mut args = env::args().skip(1);
    let mut scene_override = None;
    let mut project_root_override = None;
    let mut endpoint = BridgeEndpointConfig::default();
    let mut requested_role = BridgeControlSlot::Fighter1;
    let mut local_pilot_mode = LocalPilotMode::Human;
    let mut launcher_session_id: Option<String> = None;
    let mut launch_token: Option<String> = None;
    let mut child_kind: Option<LobbyClientKind> = None;
    let mut model_checkpoint = env::var("DFB_MODEL_CHECKPOINT").ok();
    let mut model_dataset_root = env::var("DFB_MODEL_DATASET_ROOT").ok();
    let mut model_python = env::var("DFB_MODEL_PYTHON").ok();
    let mut model_device = env::var("DFB_MODEL_DEVICE").ok();
    let mut model_observation_source = env::var("DFB_MODEL_OBSERVATION_SOURCE").ok();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scene" => {
                if let Some(scene_name) = args.next() {
                    scene_override = Some(scene_name);
                } else {
                    eprintln!("missing value for --scene");
                    std::process::exit(2);
                }
            }
            "--project-root" => {
                if let Some(project_root) = args.next() {
                    project_root_override = Some(project_root);
                } else {
                    eprintln!("missing value for --project-root");
                    std::process::exit(2);
                }
            }
            "--bridge" => {
                let Some(mode) = args.next() else {
                    eprintln!("missing value for --bridge");
                    std::process::exit(2);
                };
                endpoint.transport = match mode.as_str() {
                    "in_process" => BridgeTransport::InProcess,
                    "tcp" => BridgeTransport::Tcp,
                    other => {
                        eprintln!("unsupported bridge transport: {other}");
                        std::process::exit(2);
                    }
                };
            }
            "--bridge-addr" => {
                let Some(address) = args.next() else {
                    eprintln!("missing value for --bridge-addr");
                    std::process::exit(2);
                };
                endpoint.address = address;
            }
            "--bridge-session" => {
                let Some(session) = args.next() else {
                    eprintln!("missing value for --bridge-session");
                    std::process::exit(2);
                };
                endpoint.session = session;
            }
            "--control-role" => {
                let Some(role) = args.next() else {
                    eprintln!("missing value for --control-role");
                    std::process::exit(2);
                };
                requested_role = match role.as_str() {
                    "fighter1" | "player1" => BridgeControlSlot::Fighter1,
                    "fighter2" | "player2" => BridgeControlSlot::Fighter2,
                    "spectator" => BridgeControlSlot::Spectator,
                    other => {
                        eprintln!("unsupported control role: {other}");
                        std::process::exit(2);
                    }
                };
            }
            "--follow-ai" => {
                local_pilot_mode = LocalPilotMode::FollowAi;
            }
            "--imperfect-follow-ai" => {
                local_pilot_mode = LocalPilotMode::ImperfectFollowAi;
            }
            "--teacher-follow-ai" => {
                local_pilot_mode = LocalPilotMode::TeacherFollowAi;
            }
            "--model-control" => {
                local_pilot_mode = LocalPilotMode::Model;
            }
            "--model-checkpoint" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --model-checkpoint");
                    std::process::exit(2);
                };
                model_checkpoint = Some(value);
            }
            "--model-dataset-root" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --model-dataset-root");
                    std::process::exit(2);
                };
                model_dataset_root = Some(value);
            }
            "--model-python" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --model-python");
                    std::process::exit(2);
                };
                model_python = Some(value);
            }
            "--model-device" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --model-device");
                    std::process::exit(2);
                };
                model_device = Some(value);
            }
            "--model-observation-source" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --model-observation-source");
                    std::process::exit(2);
                };
                match value.as_str() {
                    "authoritative" | "local" | "local_predicted" => {
                        model_observation_source = Some(value);
                    }
                    other => {
                        eprintln!("unsupported model observation source: {other}");
                        std::process::exit(2);
                    }
                }
            }
            "--launcher-session-id" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --launcher-session-id");
                    std::process::exit(2);
                };
                launcher_session_id = Some(value);
            }
            "--launch-token" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --launch-token");
                    std::process::exit(2);
                };
                launch_token = Some(value);
            }
            "--child-kind" => {
                let Some(value) = args.next() else {
                    eprintln!("missing value for --child-kind");
                    std::process::exit(2);
                };
                child_kind = match value.as_str() {
                    "gameplay" => Some(LobbyClientKind::Gameplay),
                    "observer" => Some(LobbyClientKind::Observer),
                    "launcher" => Some(LobbyClientKind::Launcher),
                    other => {
                        eprintln!("unsupported child kind: {other}");
                        std::process::exit(2);
                    }
                };
            }
            _ => {}
        };
    }

    let mut app = build_client_app_with_paths(ConfigPaths {
        project_root: project_root_override
            .map(Into::into)
            .unwrap_or_else(resolve_project_root),
        scene_override,
        scene_override_path: None,
    });
    app.insert_resource(endpoint);
    app.insert_resource(RequestedControlRole(requested_role));
    app.insert_resource(local_pilot_mode);
    app.insert_resource(ModelControlConfig {
        checkpoint: model_checkpoint,
        dataset_root: model_dataset_root,
        python_executable: model_python,
        device: model_device,
        observation_source: model_observation_source,
    });
    app.insert_resource(BridgeLaunchBinding {
        launcher_session_id,
        launch_token,
        child_kind,
    });
    app.run();
}
