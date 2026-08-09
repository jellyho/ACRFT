"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_rlt as pi0_rlt
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.gr1_policy as gr1_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.policies.yam_policy as yam_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.progress as _progress
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Subset of episode indices to load (LeRobot v3 `episodes=`); None loads every episode. Used to
    # train on success-only teleop data — the factory resolves the list, so downstream (including the
    # norm-stats pass, which shares create_torch_dataset) sees exactly the trained-on episodes. The
    # filtered dataset keeps ORIGINAL episode_index values, so progress labels still line up.
    episodes: tuple[int, ...] | None = None

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRoboCasaDataConfig(DataConfigFactory):
    """Data config for RoboCasa 365 datasets converted to LeRobot v3.0.

    See ``examples/robocasa/`` for how to download the target/human demos and convert them.
    RoboCasa (PandaOmron) samples carry three cameras, a 16-d state, and a 12-d action.
    """

    # RoboCasa/robosuite actions are already delta end-effector commands (gripper absolute), so
    # no delta conversion is applied by default. Set True only if your data uses absolute actions.
    extra_delta_transform: bool = False

    # Inject a scalar `progress` label (time-to-success, from the sparse reward) into every sample.
    # Needed by Pi0RLT's progress objective; harmless but wasted work otherwise.
    include_progress: bool = False
    # Dataset column holding the sparse success signal that defines "task done" for `progress`.
    # Defaults to the LeRobot convention; a dataset without it falls back to episode-end-is-the-goal.
    reward_key: str = "next.reward"
    # Carry each frame's `episode_index` through to the model. Needed by Pi0RLT's episode-adversarial
    # objective (make z_rl un-decodable into "which demo"); train-only, dropped at inference.
    include_episode_index: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Remap the LeRobot dataset keys onto the ``observation/*`` keys read by RoboCasaInputs.
        # RepackTransform DROPS anything not listed here, so `progress` has to be mapped explicitly
        # (and produced by AddProgress, which runs first, while the raw LeRobot keys still exist).
        structure = {
            "observation/image": "observation.images.robot0_agentview_left",
            "observation/wrist_image": "observation.images.robot0_eye_in_hand",
            "observation/image_right": "observation.images.robot0_agentview_right",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        # `episode_index` is already on the raw LeRobot item, so it only has to survive the repack.
        if self.include_episode_index:
            structure["episode_index"] = "episode_index"
        repack_inputs = []
        if self.include_progress:
            structure["progress"] = "progress"
            repack_inputs.append(
                _progress.AddProgress(_progress.compute_progress_labels(self.repo_id, reward_key=self.reward_key))
            )
        repack_inputs.append(_transforms.RepackTransform(structure))
        repack_transform = _transforms.Group(inputs=repack_inputs)

        data_transforms = _transforms.Group(
            inputs=[robocasa_policy.RoboCasaInputs(model_type=model_config.model_type)],
            outputs=[robocasa_policy.RoboCasaOutputs()],
        )

        # RoboCasa actions are deltas out of the box (like Libero); only convert if configured.
        if self.extra_delta_transform:
            # Leave the discrete control-mode (idx 4) and the absolute gripper (idx 11) unchanged.
            delta_action_mask = _transforms.make_bool_mask(4, -1, 3, 3, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # RoboCasa's action column is named "action" (singular), unlike the default "actions".
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotGR1DataConfig(DataConfigFactory):
    """GR1 tabletop Teleop-Sim (LeRobot layout): one ego_view camera, 44-d state/action.

    Point HF_LEROBOT_HOME at the downloaded LeRobot root (e.g. /scratch/.../gr1_data/LeRobot) and
    use the dataset dir name as repo_id (e.g. "gr1_unified.PnPCanToDrawerClose").
    """

    include_progress: bool = False
    reward_key: str = "next.reward"
    include_episode_index: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        structure = {
            "observation/image": "observation.images.ego_view",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if self.include_episode_index:
            structure["episode_index"] = "episode_index"
        repack_inputs = []
        if self.include_progress:
            structure["progress"] = "progress"
            repack_inputs.append(
                _progress.AddProgress(_progress.compute_progress_labels(self.repo_id, reward_key=self.reward_key))
            )
        repack_inputs.append(_transforms.RepackTransform(structure))
        repack_transform = _transforms.Group(inputs=repack_inputs)

        data_transforms = _transforms.Group(
            inputs=[gr1_policy.GR1Inputs(model_type=model_config.model_type)],
            outputs=[gr1_policy.GR1Outputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # GR1's action column is also named "action" (singular).
            action_sequence_keys=("action",),
        )


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
    include_progress: bool = False
    reward_key: str = "next.reward"
    # Train on successful episodes only. The YAM teleop set is 100 success / 19 fail; with this off
    # (the default) BC clones the failures too, which is what a critic wants as negatives later but
    # is not what a clean BC policy wants. On, it resolves the success episode list from
    # outcomes.jsonl and trains (and computes norm stats) on exactly those.
    success_only: bool = False

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
        repack_inputs = []
        if self.include_progress:
            structure["progress"] = "progress"
            repack_inputs.append(
                _progress.AddProgress(_progress.compute_progress_labels(self.repo_id, reward_key=self.reward_key))
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
                    f"success_only=True but {self.repo_id} has no outcomes.jsonl and no "
                    f"'{self.reward_key}' column to derive success from."
                )
            episodes = tuple(episodes)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            episodes=episodes,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name. Used as the wandb project for every run unless overridden with --project-name.
    project_name: str = "acrft"
    # Optional wandb entity (team/user). None -> your default wandb entity.
    wandb_entity: str | None = None
    # Optional wandb group: groups related runs (e.g. one sweep) together in the wandb UI. None -> ungrouped.
    wandb_group: str | None = None
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to log action-distribution diagnostics (sampled-action spread given an
    # observation, to watch for mode collapse / overfitting). 0 disables it.
    action_dist_interval: int = 0
    # Number of action samples drawn per observation for the diagnostic above.
    action_dist_num_samples: int = 32
    # How often (in steps) to log RLT embedding-quality diagnostics (participation ratio + z stats,
    # for Pi0RLT models). 0 disables it.
    rlt_monitor_interval: int = 0
    # How often (in steps) to log the RLT embedding visualization (PCA trajectory paths + held-out
    # linear-probe R^2, host-side). 0 disables it.
    rlt_vis_interval: int = 0
    # How often (in steps) to run an in-process RoboCasa sim eval of BOTH the full VLA policy and the
    # latent BC-probe head (Pi0RLT with rlt_bc_probe=True). Headless, no video. 0 disables it.
    rlt_probe_eval_interval: int = 0
    # Rollouts per policy per probe eval.
    rlt_probe_eval_trials: int = 20
    # Base seed for the probe eval. Trial i always runs scene `seed + i` and the policy always starts
    # from the same sampling noise, so every eval — at every step, and across every run that keeps
    # this value — is scored on an identical set of episodes. Change it only to draw a different
    # (still fixed) eval set; the numbers are then not comparable to runs using the old value.
    rlt_probe_eval_seed: int = 0
    # Number of episodes drawn as trajectory paths in the RLT embedding visualization. Also sets the
    # held-out-episode probe split (half fit / half scored), so keep it >= 4.
    rlt_vis_num_trajectories: int = 8
    # Frames sampled per trajectory (evenly spaced over the episode) for that visualization.
    rlt_vis_frames_per_trajectory: int = 24
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
# Normalization stats shared by every RoboCasa 365 task, CHECKED INTO THE REPO so a fresh clone can
# train/serve without downloading any dataset just to recompute them. They are shared (not per-task)
# on purpose: per-task stats are ill-conditioned for near-stationary tasks, where a near-constant
# base/control dim gives a ~0 range that blows the loss up. `compute_shared_norm_stats.py` writes
# byte-identical files for all 50 tasks, so a single copy under a shared asset id suffices.
#
# Only the RLT configs point here. The BC configs keep asset_id=<repo_id>: serving reads norm stats
# from the checkpoint's own assets dir keyed by asset_id, so changing it would break the already
# trained pi05_robocasa_<Task> checkpoints.
_ROBOCASA_SHARED_ASSETS = AssetsConfig(assets_dir="./examples/robocasa/norm_stats", asset_id="robocasa365_shared")

_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning RoboCasa 365 configs.
    #
    # Trains on a single RoboCasa 365 target task converted to LeRobot v3.0 (see
    # examples/robocasa/). ``repo_id`` selects the dataset:
    #   - after `prepare_robocasa365.py --push-to-hub`: use the Hub id, e.g. "jellyho/robocasa365-<Task>".
    #   - to train from the local converted dir instead: set HF_LEROBOT_HOME=/data5/jellyho/robocasa365
    #     and use the bare task name as repo_id, e.g. repo_id="PickPlaceCounterToCabinet".
    # Run `scripts/compute_norm_stats.py --config-name=pi05_robocasa` before training.
    TrainConfig(
        name="pi05_robocasa",
        # action_horizon is the predicted action-chunk length (RoboCasa runs at 20 fps); tune as needed.
        model=pi0_config.Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False),
        data=LeRobotRoboCasaDataConfig(
            repo_id="jellyho/robocasa365-PrepareCoffee",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        # Constant 5e-5 after a short warmup (peak == decay), matching the pi05 LIBERO recipe.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100_000,
        save_interval=10_000,
        action_dist_interval=0,  # disabled: action_dist metric no longer logged to wandb
    ),
    # RLT ("RL Token") variant of pi05_robocasa: learns the compact RL-token bottleneck jointly with
    # the BC finetune (language-conditioned token, single forward). Same data/optimizer as pi05_robocasa;
    # the RLT loss (reconstruction + proprio) is added on top and monitored. Variant switches live on
    # Pi0RLTConfig: rlt_backbone_gradient (RLT grad into the VLM backbone), rlt_target_stop_gradient.
    # No freeze filter -> the VLA is BC-finetuned and the rlt_* bottleneck trains together.
    TrainConfig(
        name="pi05_robocasa_rlt",
        model=pi0_rlt.Pi0RLTConfig(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
            # readout head by default: RLT loss does not reshape the backbone (BC does). Flip to True to
            # let the RLT loss flow into the VLM features.
            rlt_backbone_gradient=False,
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id="jellyho/robocasa365-PrepareCoffee",
            # Repo-checked-in shared stats (see _ROBOCASA_SHARED_ASSETS): no recomputation, and no
            # dataset download needed just to normalize.
            assets=_ROBOCASA_SHARED_ASSETS,
            include_progress=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5
        ),
        # Keeps the freshly-initialized rlt_* params (absent from pi05_base) while loading the VLA.
        weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        action_dist_interval=0,  # disabled: action_dist metric no longer logged to wandb
        rlt_monitor_interval=1_000,
        rlt_vis_interval=10_000,
    ),
    # GR1 tabletop (Teleop-Sim) RLT finetune - the pilot for the GR1 port (see slurm/gr1_config_draft.py).
    # Data: HF_LEROBOT_HOME must point at the downloaded LeRobot root; repo_id is the dataset dir name.
    TrainConfig(
        name="pi05_gr1_rlt",
        model=pi0_rlt.Pi0RLTConfig(
            pi05=True,
            action_horizon=16,
            action_dim=48,  # GR1 action is 44-d; pad to 48 (pi05_base was 32 - projections re-init fresh)
            discrete_state_input=False,
            rlt_backbone_gradient=False,
        ),
        data=LeRobotGR1DataConfig(
            repo_id="gr1_unified.PnPCanToDrawerClose",
            include_progress=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,  # pilot: enough for the headroom/spread measurement
        save_interval=10_000,
        action_dist_interval=0,
        rlt_monitor_interval=1_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")


def _robocasa_task_config(task: str) -> TrainConfig:
    """A pi05 fine-tune config for a single RoboCasa 365 target task (same recipe as pi05_robocasa).

    Registered as ``pi05_robocasa_<Task>`` for every target task so norm-stats and training can be
    run per task by config name (see examples/robocasa/run_train.sh).
    """
    return TrainConfig(
        name=f"pi05_robocasa_{task}",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=16, discrete_state_input=False),
        data=LeRobotRoboCasaDataConfig(
            repo_id=f"jellyho/robocasa365-{task}",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        # Constant 5e-5 after a short warmup (peak == decay).
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=100_000,
        save_interval=10_000,
        action_dist_interval=0,  # disabled: action_dist metric no longer logged to wandb
    )


# The 50 published RoboCasa 365 target tasks (atomic + composite).
_ROBOCASA_TARGET_TASKS = (
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "CuttingToolSelection",
    "DeliverStraw",
    "GarnishPancake",
    "GatherTableware",
    "GetToastedBread",
    "HeatKebabSandwich",
    "KettleBoiling",
    "LoadDishwasher",
    "MakeIceLemonade",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PackIdenticalLunches",
    "PanTransfer",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "PortionHotDogs",
    "PreSoakPan",
    "PrepareCoffee",
    "RecycleBottlesByType",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SeparateFreezerRack",
    "SetUpCuttingStation",
    "SlideDishwasherRack",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "WaffleReheat",
    "WashFruitColander",
    "WashLettuce",
    "WeighIngredients",
)
_CONFIGS.extend(_robocasa_task_config(_t) for _t in _ROBOCASA_TARGET_TASKS)


def _robocasa_rlt_task_config(task: str) -> TrainConfig:
    """RLT variant of ``pi05_robocasa_<Task>``: learns the RL-token bottleneck during the BC finetune.

    Registered as ``pi05_robocasa_<Task>_rlt``. Identical data/optimizer recipe to the BC config, so
    the only difference is the added RLT loss (+ its monitoring). Norm stats are REUSED from the BC
    config's assets dir (RLT does not change normalization), so no recomputation is needed.

    Variant switches live on the model config and can be overridden from the CLI, e.g.
        --model.rlt-backbone-gradient   (let the RLT loss reshape the VLM backbone)
        --model.rlt-loss-weight 0.5
    """
    return TrainConfig(
        name=f"pi05_robocasa_{task}_rlt",
        model=pi0_rlt.Pi0RLTConfig(
            pi05=True,
            action_horizon=16,
            discrete_state_input=False,
            rlt_backbone_gradient=False,
            rlt_bc_probe=True,
            # RoboCasa actions are 12-d; the rest of the model's 32-d action space is zero padding
            # that the probe should not waste itself regressing (see rlt_probe_action_dim).
            rlt_probe_action_dim=12,
        ),
        data=LeRobotRoboCasaDataConfig(
            repo_id=f"jellyho/robocasa365-{task}",
            assets=_ROBOCASA_SHARED_ASSETS,
            # Cheap, and lets --model.rlt-objective switch to a progress objective without a
            # second data config.
            include_progress=True,
            # Likewise carry episode_index so --model.rlt-objective can add '+epadv' with no data change.
            include_episode_index=True,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5
        ),
        # Keeps the freshly-initialized rlt_* params (absent from pi05_base) while loading the VLA.
        weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        action_dist_interval=0,  # disabled: action_dist metric no longer logged to wandb
        rlt_monitor_interval=1_000,
        rlt_vis_interval=10_000,
        rlt_probe_eval_interval=10_000,
        rlt_probe_eval_trials=20,
    )


_CONFIGS.extend(_robocasa_rlt_task_config(_t) for _t in _ROBOCASA_TARGET_TASKS)


def _yam_rlt_config(delta_mode: str = "joint") -> TrainConfig:
    """Pi0RLT on the YAM bimanual dataset (real teleop data).

    Defaults chosen for this setting: the parallel decoder and no-proprio token (the best RLT variant
    on RoboCasa - the decoder cannot bypass the bottleneck via neighbour context, and proprio is left
    out because a downstream critic sees it directly), and a relative joint action (subtract the
    current joint position). delta_mode is joint (relative) or none (absolute); it tags the run name
    so the two do not share a checkpoint dir, and the untagged name is the relative-joint default.

    Because this is real data there is no sim: no rollout / probe-policy evaluation, and no
    behaviour-cloning probe actor either (it exists only to read out the token for the sim probe).
    The RLT bottleneck is judged by its reconstruction loss and the BC loss alone.
    """
    tag = "" if delta_mode == "joint" else f"_{delta_mode}"
    return TrainConfig(
        name=f"pi05_yam_lego_taxi{tag}_rlt",
        model=pi0_rlt.Pi0RLTConfig(
            pi05=True,
            # 30 frames at YAM's 30 fps is exactly a one-second window. The 16 this started with was
            # carried over from the RoboCasa configs and covers only 0.53 s - short for a bimanual
            # chunk, and it is also the window the RLT bottleneck has to summarise into one token.
            action_horizon=30,
            discrete_state_input=False,
            rlt_backbone_gradient=False,
            rlt_decoder_mode="parallel",  # pardec: best on RoboCasa, un-bypassable bottleneck
            rlt_include_proprio=False,  # noprop: best on RoboCasa; a critic reads proprio directly
            rlt_bc_probe=False,  # real data, no sim probe to feed - skip the probe actor entirely
            rlt_probe_action_dim=14,
        ),
        data=LeRobotYAMDataConfig(
            repo_id="jellyho/yam_lego_taxi",
            delta_mode=delta_mode,
            include_progress=False,  # success is episode-level (outcomes.jsonl), no per-frame reward
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderKeepMissing(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=100_000,
        save_interval=10_000,
        action_dist_interval=0,
        rlt_monitor_interval=1_000,
        rlt_vis_interval=10_000,
        rlt_probe_eval_interval=0,  # no sim: nothing to roll out
    )


_CONFIGS.extend(_yam_rlt_config(_m) for _m in ("joint", "none"))

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
