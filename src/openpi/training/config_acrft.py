"""Train configs for the ACRFT research line: the YAM bimanual arm.

Upstream openpi's ``config.py`` holds the configs that ship with the model release (Aloha, DROID,
LIBERO, the debug configs). Everything this project adds lives here instead, so a rebase onto a new
openpi only has to merge one file, and it is always obvious which configs are ours.

``config.py`` appends :data:`CONFIGS` to its registry (lazily, in ``_configs_dict``), so the configs
below are reachable through the usual ``config.get_config(name)`` / ``config.cli()`` and nothing else
changes. The import runs from inside that function because this module imports the dataclasses it
builds on from ``config.py``; going through the registry means either module can be imported first.
"""

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_alphaflow as pi0_alphaflow
import openpi.models.pi0_awr as pi0_awr
import openpi.models.pi0_cfgrl as pi0_cfgrl
import openpi.models.pi0_config as pi0_config
import openpi.policies.yam_policy as yam_policy
import openpi.training.advantage as _advantage
from openpi.training.config import DataConfig
from openpi.training.config import DataConfigFactory
from openpi.training.config import ModelTransformFactory
from openpi.training.config import TrainConfig
import openpi.training.optimizer as _optimizer
import openpi.training.progress as _progress
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class LeRobotYAMDataConfig(DataConfigFactory):
    """Data config for the YAM bimanual dataset (jellyho/yam_lego_taxi, LeRobot v3).

    Three cameras, a 42-d state, and a 14-d JOINT action (per arm: 6 joints + 1 gripper). The action
    delta convention is selectable, since it is the thing this dataset exists to ablate:

      joint  relative joint action: subtract the current joint position, so the target is a
             displacement from where the arm is. Grippers stay absolute.
      none   absolute joint targets, as logged.
    """

    delta_mode: str = "joint"  # joint (relative) | none (absolute)
    # Directory written by scripts/annotate_advantage.py (q_data.npy + v_data.npy). When set, every
    # sample carries `advantage` = A(s, a_chunk). Left None for plain BC, where nothing reads it.
    advantage_dir: str | None = None
    # How that advantage is normalized: "zscore" for AWR's exponential weight, "raw" for CFGRL's
    # 1{A > 0} indicator. The arm's `with_*` transform sets this together with the model, because
    # the two have to agree -- see openpi/training/advantage.py.
    advantage_norm: str = "zscore"
    # Dataset column holding the sparse success signal, used to resolve `success_only` below.
    reward_key: str = "next.reward"
    # Train on successful episodes only. The YAM teleop set is 100 success / 19 fail; with this off
    # (the default) BC clones the failures too, which is what a critic wants as negatives later but
    # is not what a clean BC policy wants. On, it resolves the success episode list from
    # the dataset's own verdict features (next.success / next.done) and trains (and computes norm
    # stats) on exactly those.
    success_only: bool = False
    # Keep the failure episodes but cut their return-to-home tails. Orthogonal to success_only, so
    # the three data conditions are reachable from the config alone:
    #   success_only=False, drop_failure_homing=True  -> failures for their task behaviour, no give-up
    #   success_only=False, drop_failure_homing=False -> every frame, give-up supervision included
    #   success_only=True                             -> no failure frames at all
    #
    # DEFAULT TRUE since 2026-09-11. A failure's homing is the operator retracting from a task that
    # is NOT done -- the one signal in the set that teaches "give up". Measured on lego, those frames
    # sit at cosine 0.956 from their nearest success-task frame while the action they teach is 3.5x
    # further away than that neighbour's: near-identical picture, opposite action. Training on it by
    # default was a footgun -- it is what you get by typing the plain config name, and a run that
    # wanted it cut had to remember a suffix. The default is now the condition we actually want, and
    # `_withhoming` names the old one for anything that has to reproduce a pre-09-11 run.
    drop_failure_homing: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        structure = {
            "observation/image": "observation.images.agentview",
            "observation/wrist_image": "observation.images.wrist_left",
            "observation/image_right": "observation.images.wrist_right",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        # `advantage` has to be mapped explicitly (RepackTransform drops anything unlisted) and
        # produced by AddAdvantage, which runs first, while the raw LeRobot `index` still exists.
        repack_inputs = []
        if self.advantage_dir is not None:
            structure["advantage"] = "advantage"
            repack_inputs.append(
                _advantage.AddAdvantage(_advantage.load_advantage(self.advantage_dir, normalize=self.advantage_norm))
            )
        repack_inputs.append(_transforms.RepackTransform(structure))
        repack_transform = _transforms.Group(inputs=repack_inputs)

        data_transforms = _transforms.Group(
            inputs=[yam_policy.YAMInputs(model_type=model_config.model_type)],
            outputs=[yam_policy.YAMOutputs()],
        )

        if self.delta_mode == "joint":
            # Subtract the current joint position from each joint action dim; grippers absolute.
            # The state's joint positions sit at 0..5 (left) and 21..26 (right); DeltaActions reads
            # state[i] for action dim i, so the mask has to align them - build it explicitly.
            ref = yam_policy.joint_delta_reference()  # action_dim -> state index, -1 = absolute
            # DeltaActions subtracts state[..., :dims] where mask is True, i.e. it assumes action dim
            # i references state dim i. YAM's right joints reference state 21.., so a plain mask does
            # not line up; a small dedicated transform handles the arbitrary reference.
            data_transforms = data_transforms.push(
                inputs=[_transforms.JointDeltaActions(ref)],
                outputs=[_transforms.JointAbsoluteActions(ref)],
            )

        model_transforms = ModelTransformFactory()(model_config)
        # Resolve the success-only episode subset now (startup), so both training and the norm-stats
        # pass see the same episodes. None => train on everything.
        episodes = None
        if self.success_only:
            episodes = _progress.success_episode_indices(self.repo_id, reward_key=self.reward_key)
            if episodes is None:
                raise ValueError(
                    f"success_only=True but {self.repo_id} carries no episode verdicts "
                    f"(next.success / next.done features) and no '{self.reward_key}' column to derive "
                    "success from. A dataset recorded before the verdict moved into the LeRobot schema "
                    "still has its outcomes.jsonl next to it: migrate it with the recorder's "
                    "`workstation/yam-data migrate-outcomes <dataset-dir>` (i2rt_rllab)."
                )
            episodes = tuple(episodes)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            episodes=episodes,
            # success_only already removes every failure frame, so cutting failure homing on top is
            # a no-op; say so rather than letting the loader do the work and find nothing.
            drop_failure_homing=self.drop_failure_homing and not self.success_only,
        )


CONFIGS: list[TrainConfig] = []


def _yam_bc_config(
    delta_mode: str = "joint",
    horizon: int = 30,
    fsdp_devices: int = 1,
    num_train_steps: int = 100_000,
    *,
    task: str = "lego_taxi",
    repo_id: str = "jellyho/yam_lego_taxi",
    wandb_entity: str | None = None,
) -> TrainConfig:
    """Plain pi05 BC finetune on a YAM bimanual teleop set -- the base config every YAM arm derives from.

    The base policy is a plain pi05 BC finetune: the critic that consumes it is trained separately
    (a patch critic), so nothing is bolted onto the policy itself.
    delta_mode joint (relative) or none (absolute); fsdp_devices>1 shards the model for multi-GPU.

    `task` and `repo_id` pick the dataset and name the config, so lego-taxi and cable-tie are the SAME
    recipe pointed at two datasets rather than two hand-written configs free to drift apart. Anything
    that defines the EXPERIMENT belongs here; only the machine's own facts (--checkpoint-base-dir,
    --num-workers) stay on the command line, so the same config name reproduces the run elsewhere.
    """
    tag = "" if delta_mode == "joint" else f"_{delta_mode}"
    hsuf = "" if horizon == 30 else f"_h{horizon}"
    return TrainConfig(
        name=f"pi05_yam_{task}{tag}{hsuf}",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=horizon, discrete_state_input=False),
        data=LeRobotYAMDataConfig(
            repo_id=repo_id,
            delta_mode=delta_mode,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        fsdp_devices=fsdp_devices,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=num_train_steps, decay_lr=5e-5
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=num_train_steps,
        save_interval=25_000,
        # Default keep_period is 5_000, which divides every saved step, so nothing is ever pruned: a
        # 500k run would hold 20 checkpoints at 42 GB each. Keep the 100k milestones plus the latest.
        keep_period=100_000,
        action_dist_interval=0,
        # Every YAM run has always gone to this wandb project, and every launcher passes it on the
        # command line; naming it here only fixes the bare `train.py <config>` case.
        project_name="yam-rlt",
        wandb_entity=wandb_entity,
    )


CONFIGS.extend(_yam_bc_config(_m, num_train_steps=500_000) for _m in ("joint", "none"))
# Chunk-length ablation for the BC policy. The adaptive-chunking work needs a base policy whose chunk
# is long enough that stopping early is a real choice; at horizon 30 the longest commitment the critic
# can score is one second.
CONFIGS.append(_yam_bc_config("joint", horizon=50, num_train_steps=500_000))

# Cable tie: the same BC recipe as lego-taxi, pointed at the cable-tie teleop set. The baseline every
# other arm on this task is compared against, and the policy to deploy on the real arm.
#
# jellyho/yam_cable_tie on the Hub: 160 episodes / 251,080 frames / 30 fps, the same three cameras as
# the lego set. It replaces rl_specialist/lerobot_cable_tie_100_clean, which resolved under an NHNHOME
# path that does not mount on this cluster and does not exist on the Hub under that name or under
# Gwanwoo/ (both checked, RepositoryNotFoundError).
#
# The success-only variant (`pi05_yam_cable_tie_success`, from _data_condition below) resolves 150 of
# the 160 episodes. It did not until 2026-09-07, and the reason is worth keeping: the verdicts landed
# on `main` on 09-02, but lerobot pins CODEBASE_VERSION = "v3.0" and resolves that TAG for every read
# (lerobot_dataset.py:83, :96), and the tag still pointed before the verdict commit. Every consumer
# saw a dataset without next.success / next.done while the Hub page showed one with them. The same was
# true of jellyho/yam_lego_taxi; its success-only runs worked only because a local cache predated the
# tag, i.e. they would not have reproduced on a clean machine. See misc/scripts/move_lerobot_v3_tag.py.
CONFIGS.append(
    _yam_bc_config(
        task="cable_tie",
        repo_id="jellyho/yam_cable_tie",
        num_train_steps=200_000,
        wandb_entity="jellyho_",
    )
)


def _yam_alphaflow_config(
    *,
    delta_mode: str = "joint",
    horizon: int = 30,
    fsdp_devices: int = 1,
    init_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base/params",
) -> TrainConfig:
    """alpha-Flow finetune of pi05 on YAM: turn the 10-step flow policy into a few/one-step one.

    This exists for offline RL, not for inference speed. Every actor-critic update has to draw an
    action from the policy, and a 10-step ODE per update is the single biggest reason RL on a VLA is
    expensive; a one-step generator makes the actor update cost one forward. Unlike distillation,
    alpha-Flow gets there by pure regression on the data -- the VLA never samples during training.

    ONE run; the curriculum is a function of the training step (state.step reaches the loss via the
    `wants_step` hook), so the whole anneal happens inside this config with no checkpoint chaining:

        progress 0 - ~0.29     alpha = 1 (sigmoid still above the 1 - 5e-3 clamp). Trajectory flow
                               matching -- the BC objective with an extra r input. The pretrained
                               pi05 IS this model at init (zero-init r MLP), so this stretch is a
                               plain BC finetune warm-up.
        progress ~0.29 - ~0.71 alpha anneals (sigmoid, gamma 25, centred mid-run). Consistency
                               targets ramp up as flow matching hands over.
        progress ~0.71 - 1     alpha = 5e-3 (the clamp floor). Discrete MeanFlow-limit training;
                               the JVP branch stays off (pi0_alphaflow.py explains that default).

    This is the OFFICIAL alpha-Flow schedule (sigmoid over the whole run; the clamps carve the
    phases), in progress fractions: train.py passes progress = step / num_train_steps into the
    loss, so overriding --num-train-steps rescales the whole curriculum with it.

    The alpha = 1 stretch pays for the second (stop-gradient, no-backward) forward it does not yet
    need -- that is the price of a single run, ~15% of a BC step, and it buys not having to chain
    checkpoints or split wandb curves across runs. Watch `alpha` and `delta2` in wandb: the loss
    itself sits pinned near 1.0 by MeanFlow's adaptive weight and is NOT the progress signal.

    An optional pure-MeanFlow (JVP) polish stays available as a followup run via
    Pi0AlphaFlowConfig(alpha_init=0, alpha_final=0, meanflow_jvp=True) pointed at this run's output;
    it is deliberately not registered as a config until the discrete floor proves insufficient.
    """
    tag = "" if delta_mode == "joint" else f"_{delta_mode}"
    hsuf = "" if horizon == 30 else f"_h{horizon}"
    steps = 200_000  # schedule is in progress fractions, so the curriculum stretches with this
    return TrainConfig(
        name=f"pi05_yam_lego_taxi{tag}{hsuf}_alphaflow",
        # Schedule defaults ARE the official alpha-Flow recipe (sigmoid over the whole run, gamma
        # 25, clamp 5e-3, fm_ratio 0.5), expressed in progress fractions so --num-train-steps
        # rescales the curriculum -- nothing here to keep in sync.
        model=pi0_alphaflow.Pi0AlphaFlowConfig(
            pi05=True,
            action_horizon=horizon,
            discrete_state_input=False,
        ),
        data=LeRobotYAMDataConfig(
            repo_id="jellyho/yam_lego_taxi",
            delta_mode=delta_mode,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        fsdp_devices=fsdp_devices,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=steps, decay_lr=5e-5),
        weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(init_checkpoint),
        num_train_steps=steps,
        save_interval=10_000,
        action_dist_interval=0,
    )


CONFIGS.append(_yam_alphaflow_config())


# The artefacts both extraction arms refine and score against. They are ABSOLUTE because they name
# artefacts, not repo files, and they point at the old slurm cluster's filesystem -- on any other
# machine they have to be repointed here. The config still loads either way; the failure surfaces at
# training start as a missing checkpoint, not as a bad run. Repoint here rather than on the command
# line: the config name is the experiment.
_ADVANTAGE_DIR = "/NHNHOME/jellyho/jellyho/ACRFT/.scratch/extraction/advantage_fixed_tau9min"
_BC_CHECKPOINT = "/data5/jellyho/ACRFT/openpi/checkpoints/pi05_yam_lego_taxi/yam_bc_s300_h30_successonly/200000/params"


def with_cfgrl(
    base: TrainConfig,
    *,
    advantage_dir: str,
    init_checkpoint: str,
    cfg_w: float = 1.5,
    num_train_steps: int = 30_000,
    suffix: str = "cfgrl",
) -> TrainConfig:
    """CFGRL variant OF ANY pi0.5 task config — a method transform, not a task config.

    Policy-extraction methods are orthogonal to the task: the same CFGRL recipe should apply to
    YAM, LIBERO or anything added later, so it is expressed as base -> variant rather
    than copied per task. Everything task-shaped (data config, norm stats, horizon, action dim,
    optimizer) is inherited from `base`; only the model class changes, because the optimality
    embedding and guided sampling are model properties (kvfrans/cfgrl iql_diffusion.py:105,213).

        CONFIGS.append(with_cfgrl(_yam_bc_config()))            # pi05_yam_lego_taxi_cfgrl
        CONFIGS.append(with_cfgrl(some_libero_cfg, cfg_w=3.0))  # ... and any other task

    The other weight-bearing extraction arms (awr, flowdpg, qam, dql) need no variant at all: they
    fine-tune the plain pi0.5 action expert, so an exported checkpoint is served by the BASE config
    unchanged. CFGRL is the one arm whose network differs.
    """
    if not isinstance(base.model, pi0_config.Pi0Config):
        raise TypeError(f"with_cfgrl needs a pi0.5 base config, got {type(base.model).__name__}")
    return dataclasses.replace(
        base,
        name=f"{base.name}_{suffix}",
        model=pi0_cfgrl.Pi0CFGRLConfig(
            **{f.name: getattr(base.model, f.name) for f in dataclasses.fields(base.model)},
            cfg_w=cfg_w,
        ),
        # RAW advantage: the label is the hard indicator 1{A > 0} (iql_diffusion.py:157), so a
        # z-score would retarget the threshold to 1{A > mean(A)} without anything to notice it by.
        data=dataclasses.replace(base.data, advantage_dir=advantage_dir, advantage_norm="raw"),
        weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(init_checkpoint),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0, peak_lr=5e-5, decay_steps=num_train_steps, decay_lr=5e-5
        ),
        ema_decay=None,
        num_train_steps=num_train_steps,
        save_interval=10_000,
        keep_period=10_000,
    )


def with_awr(
    base: TrainConfig,
    *,
    advantage_dir: str,
    init_checkpoint: str,
    temp: float = 1.0,
    weight_clip: float = 20.0,
    num_train_steps: int = 30_000,
    suffix: str = "awr",
) -> TrainConfig:
    """AWR variant OF ANY pi0.5 task config -- a method transform, like `with_cfgrl`.

    Unlike CFGRL the network is unchanged, so this swaps only the loss (Pi0AWRConfig) and the three
    things that make it an EXTRACTION run rather than a BC run:

      * it starts from a finished BC checkpoint, not pi05_base;
      * it trains the action expert only (Pi0AWRConfig.get_freeze_filter), so every arm gets the
        same budget on the same frozen features;
      * its data carries the advantage annotation the loss weights by.

    Those three used to be argparse flags in scripts/train_awr.py, which meant the run was defined
    by what someone typed. Here the config name is the experiment -- and the arm gets --resume,
    FSDP, checkpoint pruning and norm-stats resolution from the ordinary trainer for free.
    """
    if not isinstance(base.model, pi0_config.Pi0Config):
        raise TypeError(f"with_awr needs a pi0.5 base config, got {type(base.model).__name__}")
    return dataclasses.replace(
        base,
        name=f"{base.name}_{suffix}",
        model=pi0_awr.Pi0AWRConfig(
            **{f.name: getattr(base.model, f.name) for f in dataclasses.fields(base.model)},
            awr_temp=temp,
            awr_weight_clip=weight_clip,
        ),
        data=dataclasses.replace(base.data, advantage_dir=advantage_dir),
        # Start from the BC policy this arm refines, keeping anything the base checkpoint lacks.
        weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(init_checkpoint),
        # The arms ran a flat 5e-5 with no EMA; the BC default (cosine + EMA 0.99) would confound
        # "the objective helped" with "the schedule helped".
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0, peak_lr=5e-5, decay_steps=num_train_steps, decay_lr=5e-5
        ),
        ema_decay=None,
        num_train_steps=num_train_steps,
        save_interval=10_000,
        keep_period=10_000,
    )


CONFIGS.append(
    with_cfgrl(
        _yam_bc_config(),
        advantage_dir=_ADVANTAGE_DIR,
        init_checkpoint=_BC_CHECKPOINT,
    )
)

# The AWR extraction arm on the deployed YAM BC policy.
CONFIGS.append(
    with_awr(
        _yam_bc_config(),
        advantage_dir=_ADVANTAGE_DIR,
        init_checkpoint=_BC_CHECKPOINT,
    )
)


def _data_condition(base: TrainConfig, suffix: str, doc: str, **data_kwargs) -> TrainConfig:
    """A named config for one episode/frame condition, derived from `base`.

    The condition belongs in the config, not in a flag on the command line, because a flag can be
    forgotten and a config name cannot. That is not hypothetical: the deployed BC policy is called
    `yam_bc_s300_h30_successonly` and trained on ALL 347 episodes. wandb kept the argv --
    `pi05_yam_lego_taxi --exp-name yam_bc_s300_h30_successonly --overwrite` -- with no
    --data.success-only anywhere on it, and its h50 sibling was typed the same way fifteen minutes
    later. Two runs, months of downstream work, one missing flag. Naming the condition in the config
    makes that mistake impossible to make silently: `--exp-name` is a label, `--config-name` is the
    experiment.

    Each variant also gets its own `assets_dirs` and `checkpoint_dir` for free (both derive from the
    config name), so two conditions can never share a norm-stats file or a checkpoint directory by
    accident. Norm stats still resolve themselves: the content cache finds an existing asset computed
    on this exact episode set wherever it was filed, and computes one only on a genuine miss.
    """
    return dataclasses.replace(base, name=f"{base.name}_{suffix}", data=dataclasses.replace(base.data, **data_kwargs))


# The YAM data conditions, as configs rather than as flags. Only the two BC bases that experiments
# actually compare are expanded -- the h50 and absolute-action variants would double the registry for
# combinations nothing runs.
for _base_name in ("pi05_yam_lego_taxi", "pi05_yam_cable_tie"):
    _base = next(c for c in CONFIGS if c.name == _base_name)
    CONFIGS.append(
        _data_condition(
            _base,
            "success",
            "successful episodes only",
            success_only=True,
        )
    )
    # The give-up condition is the DEFAULT now (see LeRobotYAMDataConfig.drop_failure_homing), so
    # the arm that needs a name is the other one: every frame, give-up supervision included. That is
    # what every run before 2026-09-11 trained on, including the deployed policy, so it has to stay
    # reachable by name -- a comparison against those runs is a comparison against this condition.
    CONFIGS.append(
        _data_condition(
            _base,
            "withhoming",
            "all episodes, failure give-up tails KEPT (the pre-2026-09-11 default)",
            drop_failure_homing=False,
        )
    )
