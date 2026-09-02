import json

from dfb_game_py import Environment, EnvironmentAction


def main() -> None:
    env = Environment(
        project_root=".",
        scene_name="open",
        seed=7,
        enable_visual=False,
        enable_audio=False,
        ticks_per_step=2,
        self_play=False,
    )
    try:
        reset = json.loads(env.reset_json(scene_name="open", seed=7, ticks_per_step=2))
        print("reset keys:", sorted(reset.keys()))

        action = EnvironmentAction(
            throttle=0.25,
            pitch=0.1,
            roll=0.0,
            yaw=0.0,
            fire_gun=False,
            repair=False,
        )
        step = json.loads(env.step_json(action))
        print("step keys:", sorted(step.keys()))

        observation = json.loads(env.latest_observation_json())
        print("observation keys:", sorted(observation.keys()))
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
