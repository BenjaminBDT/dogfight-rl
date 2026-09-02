from __future__ import annotations

import argparse
from pathlib import Path

from dfb_reinforcement_learning.scenes import ScenePoolSpec, materialize_scene_pool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tactical .ron scenes from a scene-pool JSON spec.")
    parser.add_argument("--scene-pool-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = ScenePoolSpec.from_json(Path(args.scene_pool_json))
    materialize_scene_pool(spec=spec, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
