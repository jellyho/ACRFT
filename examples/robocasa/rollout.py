"""Shared RoboCasa rollout utilities for both out-of-process (main.py, websocket) and in-process
(training-time probe eval) evaluation.

A rollout is: reset the env with a fixed per-trial seed (so every checkpoint sees identical scenes),
then repeatedly query a policy for an action chunk and execute its first ``replan_steps`` actions
until success or the horizon. The only thing that differs between call sites is the ``policy_fn``:
a websocket client for main.py, or a direct model call for inline eval.
"""

import numpy as np

# obs keys that make up the 16-d ``observation.state`` (LeRobot order), from robocasa's
# LEROBOT_STATE_TO_HDF5_STATE mapping.
STATE_KEYS = (
    "robot0_base_pos",  # 3  base position
    "robot0_base_quat",  # 4  base rotation
    "robot0_base_to_eef_pos",  # 3  EE position (relative)
    "robot0_base_to_eef_quat",  # 4  EE rotation (relative)
    "robot0_gripper_qpos",  # 2  gripper qpos
)
CAMERAS = {
    "observation/image": "robot0_agentview_left",
    "observation/wrist_image": "robot0_eye_in_hand",
    "observation/image_right": "robot0_agentview_right",
}


def state_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in STATE_KEYS])


def image_from_obs(obs: dict, camera: str) -> np.ndarray:
    # Raw robosuite camera obs are bottom-origin; RoboCasa flips them vertically to the upright
    # frames the datasets (and thus the policy) were trained on. Match that here.
    return np.ascontiguousarray(obs[f"{camera}_image"][::-1])


def lerobot_action_to_env(action: np.ndarray) -> np.ndarray:
    """Reorder a 12-d model action (LeRobot order) into the env's expected flat order.

    LeRobot order : [base_motion(0:4), control_mode(4:5), ee_pos(5:8), ee_rot(8:11), gripper(11:12)]
    Env/HDF5 order: [ee_pos(0:3), ee_rot(3:6), gripper(6:7), base_motion(7:11), control_mode(11:12)]
    """
    return np.concatenate([action[5:8], action[8:11], action[11:12], action[0:4], action[4:5]])


def make_env(task: str, *, camera_size: int = 256, seed: int = 0):
    """Create the RoboCasa target-split env used for evaluation (heldout layouts/objects)."""
    from robocasa.utils.env_utils import create_env

    return create_env(
        env_name=task,
        robots="PandaOmron",
        split="target",
        camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
        camera_widths=camera_size,
        camera_heights=camera_size,
        seed=seed,
    )


def obs_to_element(obs: dict, prompt: str) -> dict:
    """Assemble the model-input dict from a raw env observation."""
    return {
        "observation/state": state_from_obs(obs),
        "observation/image": image_from_obs(obs, CAMERAS["observation/image"]),
        "observation/wrist_image": image_from_obs(obs, CAMERAS["observation/wrist_image"]),
        "observation/image_right": image_from_obs(obs, CAMERAS["observation/image_right"]),
        "prompt": prompt,
    }


def stage_flags(env):
    """Task-stage predicates for the failure taxonomy. PrepareCoffee-shaped; returns {} elsewhere.

    grasped   the gripper currently holds the mug
    placed    the mug sits under the dispenser well enough to pour
    machine_on the start button has been pressed
    """
    try:
        from robocasa.utils import object_utils as ou

        return {
            "grasped": bool(ou.check_obj_grasped(env, "obj")),
            "placed": bool(env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj")),
            "machine_on": bool(env.coffee_machine._turned_on),
        }
    except Exception:
        return {}


def run_trials(
    env,
    policy_fn,
    *,
    task: str,
    num_trials: int,
    seed: int = 0,
    replan_steps: int = 10,  # = the model's action_horizon (chunk size); execute a full chunk per replan
    max_steps: int | None = None,
    on_trial=None,
    on_step=None,
    on_transition=None,
    ep_metas=None,
) -> dict:
    """Roll out ``policy_fn`` for ``num_trials`` and return {successes, num_trials, success_rate, trials}.

    ``policy_fn(element)`` returns either an action chunk (np array [chunk, 12], raw LeRobot action
    space) or ``(chunk, num_to_execute, info)``. The second form is what adaptive chunking needs: a
    critic that scores every prefix picks how far to commit, so the executed length varies per replan
    and cannot come from a fixed ``replan_steps``. ``info`` is passed straight through to ``on_step``
    and is never interpreted here.

    ``on_trial(trial_index, success, steps)`` fires once per trial (e.g. logging).
    ``on_transition(obs, action, step)`` fires BEFORE each env.step with the action about to run —
    the (state, action) pairs a dataset collector wants.
    ``ep_metas`` is an optional list of robocasa ep_meta dicts; trial i replays scene i (mod len)
    exactly instead of sampling a fresh one from (seed, trial).
    ``on_step(obs, info, step)`` fires after every env step, which is where a caller renders frames.
    """
    max_steps = max_steps or int(getattr(env, "horizon", 500))
    successes, trials = 0, []
    for trial in range(num_trials):
        # Deterministic per-trial scene: trial i has to depend only on (seed, trial), never on how
        # much RNG the preceding trials happened to consume - otherwise a mode whose rollouts run
        # longer sees different scenes, and two policies are no longer compared on the same task.
        # env.rng covers the samplers that take it explicitly; some robosuite placement samplers read
        # the legacy global RNG instead, so both are pinned here rather than once per eval.
        env.rng = np.random.default_rng(seed + trial)
        np.random.seed(seed + trial)
        if ep_metas is not None:
            # Replay an exact recorded scene: robocasa Kitchen envs rebuild the kitchen (layout,
            # style, every object placement) from a stored ep_meta on the next reset. This is how
            # the training demos' scenes are put back in front of the policy - the in-distribution
            # evaluation that fresh seeds cannot give.
            env.set_ep_meta(ep_metas[trial % len(ep_metas)])
        obs = env.reset()
        prompt = env.get_ep_meta().get("lang", task)
        success, step = False, 0
        stage_at: dict = {}
        while step < max_steps and not success:
            out = policy_fn(obs_to_element(obs, prompt))
            info = None
            if isinstance(out, tuple):
                chunk, n_exec, *rest = out
                info = rest[0] if rest else None
            else:
                chunk, n_exec = out, replan_steps
            chunk = np.asarray(chunk)
            for action in chunk[: max(int(n_exec), 1)]:
                for k, v in stage_flags(env).items():
                    if v and k not in stage_at:
                        stage_at[k] = step  # first step each stage was observed true
                if on_transition is not None:
                    # Fires with the PRE-step obs: (obs_t, action executed at t) is the pair an
                    # annotation pass needs; post-step obs would shift every frame by one.
                    on_transition(obs, np.asarray(action), step)
                obs, _, _, _ = env.step(lerobot_action_to_env(np.asarray(action)))
                step += 1
                if on_step is not None:
                    on_step(obs, info, step)
                if env._check_success():
                    success = True
                    break
                if step >= max_steps:
                    break
        successes += int(success)
        meta = env.get_ep_meta() if hasattr(env, "get_ep_meta") else {}
        trials.append(
            {
                "trial": trial,
                "success": bool(success),
                "steps": step,
                # Scene fingerprint: lets any later audit confirm two runs really replayed the same
                # scenes without keeping the sim around.
                "layout": meta.get("layout_id"),
                "style": meta.get("style_id"),
                # First step each task stage was reached (absent = never): the failure taxonomy
                # reads straight off this - no grasp / grasped-but-not-placed / placed-but-no-press.
                "stage_at": stage_at,
            }
        )
        if on_trial is not None:
            on_trial(trial, success, step)
    return {
        "successes": successes,
        "num_trials": num_trials,
        "success_rate": successes / max(num_trials, 1),
        "trials": trials,
    }
