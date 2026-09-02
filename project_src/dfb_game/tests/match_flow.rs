use dfb_game::core::config::GameConfig;

#[test]
fn default_match_time_limit_is_unbounded() {
    assert_eq!(GameConfig::default().match_time_limit_seconds, None);
}
