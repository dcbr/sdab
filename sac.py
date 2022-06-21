from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
import gym

from stable_baselines3.common import logger
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)
from stable_baselines3.sac.sac import SAC as baseSAC
from stable_baselines3.sac.policies import Actor as baseSACActor, SACPolicy as baseSACPolicy


class SAC(baseSAC):
    """SAC class without SDE and logging to Tensorboard."""

    def __init__(
        self,
        policy: Union[str, Type["SACPolicy"]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq,
                         gradient_steps, action_noise, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage,
                         ent_coef, target_update_interval, target_entropy, False, -1, False, tensorboard_log,
                         create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        self.set_logger(logger.configure(self.tensorboard_log, ["stdout", "csv", "tensorboard"]))
        return super()._setup_learn(total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, log_path,
                                    reset_num_timesteps, tb_log_name)


class SACActor(baseSACActor):
    """SACActor class without SDE."""

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            **kwargs
    ):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn,
                         False, -3, True, None, False, 2.0, normalize_images)


class SACPolicy(baseSACPolicy):
    """SACPolicy class without SDE and tracking of distribution parameters."""

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Box,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, False, -3,
                         None, False, 2.0, features_extractor_class, features_extractor_kwargs,
                         normalize_images, optimizer_class, optimizer_kwargs, n_critics, share_features_extractor)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SACActor:
        # Return SACActor instead of baseSACActor
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return SACActor(**actor_kwargs).to(self.device)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        actions = super()._predict(observation, deterministic)
        # Store action distribution parameters for later analysis
        with torch.no_grad():
            mean, log_std, *_ = self.actor.get_action_dist_params(observation)
        self._last_mean, self._last_std = mean.cpu().numpy(), log_std.exp().cpu().numpy()
        return actions
