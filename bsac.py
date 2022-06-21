from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
import numpy as np
import torch
from torch import nn
import gym

from stable_baselines3.sac.policies import LOG_STD_MIN, LOG_STD_MAX
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)

from sac import SAC, SACPolicy, SACActor
from trunc_norm import TruncGaussianDistribution
import rescaling


class BSAC(SAC):
    """SAC with support for custom action bounds."""

    def __init__(
        self,
        policy: Union[str, Type["BSACPolicy"]],
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
                         ent_coef, target_update_interval, target_entropy, tensorboard_log, create_eval_env,
                         policy_kwargs, verbose, seed, device, _init_setup_model)
        self.rng = np.random.default_rng(seed)

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None, n_envs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            # Warmup phase
            norm_action = self.rng.uniform(-1, 1, (n_envs, get_action_dim(self.action_space)))
            action = self.policy.rescale_action(norm_action, self._last_obs, False)
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            action, _ = self.predict(self._last_obs, deterministic=False)
            norm_action = self.policy._last_norm_action

        # We store the normalized action in the buffer
        return action, norm_action


class BSACActor(SACActor):
    """SACActor with support for custom action bounds."""

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
                         normalize_images)
        action_dim = get_action_dim(self.action_space)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        # Use truncated Gaussian distribution and use tanh as activation for the last mu layer
        self.action_dist = TruncGaussianDistribution(action_dim)
        self.mu = nn.Sequential(nn.Linear(last_layer_dim, action_dim), nn.Tanh())
        self.log_std = nn.Linear(last_layer_dim, action_dim)

    def get_action_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Returns the mean, log standard deviation, lower bound and upper bound parameters of the truncated Gaussian distribution
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {"low": torch.tensor(-1), "high": torch.tensor(1)}

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Sanity check to make sure the actor's rescaling functions are not used.")

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Sanity check to make sure the actor's rescaling functions are not used.")


class BSACPolicy(SACPolicy):
    """SACPolicy with support for custom action bounds."""

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
            bounds: Optional[Union[List, Callable]] = None,
            rescale: str = "lin",
            **kwargs
    ):
        """
        :param bounds: Defines the lower and upper bounds for each action component. Can be
                        either a list of static bounds (constant for each observation) or a
                        mapping (callable) from observations to a list of bounds (variable
                        for each observation). If set to ``None``, static bounds are used
                        as determined by the action_space.
        :param rescale: The tag of the rescaling function to use.
        """
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn,
                         features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class,
                         optimizer_kwargs, n_critics, share_features_extractor)
        p_max = rescaling.Params(action_space.low, action_space.high)
        self._rescaling = rescaling.from_tag(rescale, p_max=p_max)
        if bounds is None:
            bounds = [action_space.low, action_space.high]
        self._bounds = bounds if isinstance(bounds, Callable) else lambda obs: bounds

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> BSACActor:
        # Return BSACActor instead of SACActor
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return BSACActor(**actor_kwargs).to(self.device)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        norm_actions = super()._predict(observation, deterministic)

        # Convert to numpy
        norm_actions = norm_actions.cpu().numpy()
        obs = observation.cpu().numpy()

        # Rescale to proper domain
        actions = self.rescale_action(norm_actions, obs, False)

        # Store action bounds and normalized actions for later usage and analysis
        self._last_bounds = np.reshape(self.get_bounds(obs), (2, -1))
        self._last_norm_action = norm_actions
        return torch.as_tensor(actions)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Sanity check to make sure the policy's rescaling functions are not used.")

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        # raise NotImplementedError("Sanity check to make sure the policy's rescaling functions are not used.")
        # This is called once in the predict function. Just return the scaled_action here, as we deal with
        # the rescaling already in the _predict function.
        return scaled_action

    def get_bounds(self, obs: np.ndarray) -> np.ndarray:
        # Retrieve the action bounds for the given observation
        bounds = self._bounds(obs)
        return np.clip(bounds, self.action_space.low, self.action_space.high)  # Always respect action space bounds

    def rescale_action(self, action: np.ndarray, obs: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Rescale the (normalized) action from [-1, 1] to the state-dependent bounds [l(s), u(s)].
        Or the inverse scaling operation when inverse is set to True.
        """
        bounds = self.get_bounds(obs)
        rp = rescaling.Params(bounds[0], bounds[1])
        return self._rescaling.inverse(action, rp) if inverse else self._rescaling.rescale(action, rp)
