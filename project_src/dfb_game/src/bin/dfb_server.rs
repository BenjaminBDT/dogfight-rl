use dfb_game::app::game_app::{ServerTimeMode, build_server_app_with_paths_and_admin};
use dfb_game::bridge::ipc_stub::{BridgeEndpointConfig, BridgeTransport};
use dfb_game::core::config::{ConfigPaths, resolve_project_root};
use dfb_game::gameplay::server_admin::{ServerAdminInterfaceMode, default_interface_mode};
use dfb_game::recording::ServerAuthoritativeRecording;
use std::env;
use std::io::IsTerminal;

fn main() {
    let mut args = env::args().skip(1);
    let mut scene_override = None;
    let mut time_mode = ServerTimeMode::Realtime;
    let mut record_authoritative = false;
    let mut admin_interface = default_interface_mode();
    let mut endpoint = BridgeEndpointConfig {
        transport: BridgeTransport::Tcp,
        ..Default::default()
    };

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
            "--time-mode" => {
                let Some(mode) = args.next() else {
                    eprintln!("missing value for --time-mode");
                    std::process::exit(2);
                };
                time_mode = match mode.as_str() {
                    "realtime" => ServerTimeMode::Realtime,
                    "simulated" => ServerTimeMode::Simulated,
                    other => {
                        eprintln!("unsupported time mode: {other}");
                        std::process::exit(2);
                    }
                };
            }
            "--record-authoritative" => {
                record_authoritative = true;
            }
            "--admin-ui" => {
                let Some(mode) = args.next() else {
                    eprintln!("missing value for --admin-ui");
                    std::process::exit(2);
                };
                admin_interface = match mode.as_str() {
                    "off" => ServerAdminInterfaceMode::Off,
                    "plain" => ServerAdminInterfaceMode::Plain,
                    "tui" => ServerAdminInterfaceMode::Tui,
                    "auto" => {
                        if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
                            ServerAdminInterfaceMode::Tui
                        } else {
                            ServerAdminInterfaceMode::Plain
                        }
                    }
                    other => {
                        eprintln!("unsupported admin ui mode: {other}");
                        std::process::exit(2);
                    }
                };
            }
            _ => {}
        };
    }

    let mut app = build_server_app_with_paths_and_admin(
        ConfigPaths {
            project_root: resolve_project_root(),
            scene_override,
            scene_override_path: None,
        },
        time_mode,
        admin_interface,
    );
    app.insert_resource(endpoint);
    app.insert_resource(ServerAuthoritativeRecording {
        enabled: record_authoritative,
    });
    app.run();
}
