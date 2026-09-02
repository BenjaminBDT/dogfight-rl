use dfb_game::bridge::BridgeSmoothingTuning;

#[test]
fn bridge_smoothing_defaults_are_in_sane_ranges() {
    let tuning = BridgeSmoothingTuning::default();

    assert!(tuning.snapshot_buffer_fast_consume_lead_ticks > 0.0);
    assert!(tuning.snapshot_buffer_slow_consume_lead_ticks > 0.0);
    assert!(
        tuning.snapshot_buffer_fast_consume_lead_ticks
            > tuning.snapshot_buffer_slow_consume_lead_ticks
    );

    assert!(
        tuning.remote_snapshot_position_blend > 0.0 && tuning.remote_snapshot_position_blend < 1.0
    );
    assert!(
        tuning.remote_snapshot_rotation_blend > 0.0 && tuning.remote_snapshot_rotation_blend < 1.0
    );
    assert!(
        tuning.remote_snapshot_velocity_blend > 0.0 && tuning.remote_snapshot_velocity_blend < 1.0
    );
    assert!(
        tuning.remote_snapshot_angular_blend > 0.0 && tuning.remote_snapshot_angular_blend < 1.0
    );

    assert!(tuning.local_correction_unacked_factor > 0.0);
    assert!(tuning.local_correction_unacked_factor < tuning.local_correction_acked_factor);
    assert!(
        tuning.local_correction_rotation_alpha_unacked_max
            < tuning.local_correction_rotation_alpha_acked_max
    );
    assert!(
        tuning.local_correction_velocity_alpha_unacked_max
            < tuning.local_correction_velocity_alpha_acked_max
    );
    assert!(
        tuning.local_correction_angular_alpha_unacked_max
            < tuning.local_correction_angular_alpha_acked_max
    );
}
