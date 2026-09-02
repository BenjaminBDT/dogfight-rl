use dfb_game::core::config::RepositoryConfig;
use std::path::Path;

#[test]
fn scene_override_updates_active_scene_and_loads_open_layout() {
    let config =
        RepositoryConfig::load_from_root_with_scene("../..", Some("open")).expect("load open");

    assert_eq!(config.game.active_scene, "open");
    assert_eq!(config.scene.obstacles.len(), 0);
}

#[test]
fn scene_override_updates_active_scene_and_loads_default_layout() {
    let config = RepositoryConfig::load_from_root_with_scene("../..", Some("default"))
        .expect("load default");

    assert_eq!(config.game.active_scene, "default");
    assert!(!config.scene.obstacles.is_empty());
}

#[test]
fn new_open_variants_load_successfully() {
    for scene_name in [
        "open_head_on_200m",
        "open_fighter1_tail_chase",
        "open_fighter2_tail_chase",
        "open_fighter1_side_cut_in",
        "open_fighter2_side_cut_in",
    ] {
        let config = RepositoryConfig::load_from_root_with_scene("../..", Some(scene_name))
            .unwrap_or_else(|error| panic!("load {scene_name}: {error:#}"));
        assert_eq!(config.game.active_scene, scene_name);
        assert_eq!(config.scene.obstacles.len(), 0);
    }
}

#[test]
fn scene_path_override_loads_named_scene_file() {
    let config = RepositoryConfig::load_from_root_with_scene_path(
        "../..",
        Some(Path::new(
            "../../config/dfb_game/scenes/open_head_on_200m.ron",
        )),
    )
    .expect("load scene path");

    assert_eq!(config.game.active_scene, "open_head_on_200m");
    assert_eq!(config.scene.obstacles.len(), 0);
}
