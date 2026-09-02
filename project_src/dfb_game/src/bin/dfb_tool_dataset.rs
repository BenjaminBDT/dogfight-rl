use anyhow::{Result, bail};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        bail!("missing subcommand");
    };

    match command.as_str() {
        "pack" => dfb_game::dataset_tool::pack::run_from_args(args),
        "pack-part3-bc" => dfb_game::dataset_tool::part3_bc::run_from_args(args),
        "part3-policy-observation-fixture" => {
            dfb_game::dataset_tool::part3_policy::run_fixture_from_args(args)
        }
        "trim-recording" => dfb_game::dataset_tool::recording_trim::run_from_args(args),
        "extract" | "reconstruct" => dfb_game::dataset_tool::extract::run_from_args(args),
        "label" => dfb_game::dataset_tool::label::run_from_args(args),
        "audit-visibility" => dfb_game::dataset_tool::label::run_visibility_audit_from_args(args),
        "synthetic-single-step" => {
            dfb_game::dataset_tool::label::run_synthetic_single_step_from_args(args)
        }
        "synthetic-visual" => dfb_game::dataset_tool::synthetic_visual::run_from_args(args),
        "--help" | "-h" | "help" => {
            print_help();
            Ok(())
        }
        other => {
            print_help();
            bail!("unknown subcommand: {other}")
        }
    }
}

fn print_help() {
    eprintln!(
        "Usage:\n  dfb_tool_dataset pack [pack args]\n  dfb_tool_dataset pack-part3-bc [part3 bc args]\n  dfb_tool_dataset part3-policy-observation-fixture --fixture <path>\n  dfb_tool_dataset trim-recording --episode <episode-dir> --end-step <inclusive-step> [--reason <reason>]\n  dfb_tool_dataset extract [extract args]\n  dfb_tool_dataset reconstruct [extract args]\n  dfb_tool_dataset label [label args]\n  dfb_tool_dataset audit-visibility [audit args]\n  dfb_tool_dataset synthetic-single-step [synthetic single-step args]\n  dfb_tool_dataset synthetic-visual [synthetic args]\n\nCurrent status:\n  - pack: exports Part 2 physical dataset layout as chunked npz + meta.json + schema.json\n  - pack-part3-bc: exports Part 3 policy-contract behavior-cloning dataset as chunked npz + meta/schema/normalizer\n  - part3-policy-observation-fixture: emits Rust policy observations for parity tests\n  - trim-recording: transactionally trims a recording to an inclusive step and preserves a local backup\n  - extract/reconstruct: reconstructs per-role multimodal observations from authoritative recordings\n  - label: derives GT-relative state labels, audio structure features, and heuristic rule targets\n  - audit-visibility: diagnoses segmentation/depth/keypoint visibility agreement before training\n  - synthetic-single-step: derives keypoint / sparse voting / geometry labels from synthetic clean roots\n  - synthetic-visual: generates a small synthetic visual prototype set for single-step vision recovery"
    );
}
