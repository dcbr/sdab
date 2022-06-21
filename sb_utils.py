from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices
from stable_baselines3.common.callbacks import CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.type_aliases import Schedule

import pathlib
import numpy as np
import gym
from PIL import Image, ImageDraw, ImageOps, ImageFont

from typing import Any, List, Tuple, Optional, Sequence, Type, Union


def create_noise(noise_cfg, action_dim):
    """Create noise instance from configuration dict."""
    if noise_cfg["type"] == 'normal':
        return NormalActionNoise(noise_cfg.get("mean", 0)*np.ones(action_dim), noise_cfg.get("std", 1)*np.ones(action_dim))
    elif noise_cfg["type"] == 'ou':
        return OrnsteinUhlenbeckActionNoise(noise_cfg.get("mean", 0)*np.ones(action_dim), noise_cfg.get("std", 1)*np.ones(action_dim))
    else:
        return None


class MetricWrapper(gym.Wrapper):
    """Base class for environment wrappers which measure certain metrics at every timestep.
    These metrics will be aggregated and logged by the custom EvalMetricCallback."""
    AGGREGATIONS = {}
    # Dictionary mapping metric names to aggregation modes ('min', 'max', 'mean', 'std', 'sum', 'raw').

    def __init__(self, env):
        super().__init__(env)
        # Extend aggregations dictionary and attach it to the environment
        aggrs = getattr(env, "aggrs", {})
        aggrs.update(self.AGGREGATIONS)
        env.aggrs = aggrs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Retrieve the metrics for the current timestep and put them in the info dict.
        metrics = info.get("metrics", {})
        metrics.update(self.metrics(action, obs, reward, done))
        info["metrics"] = metrics
        return obs, reward, done, info

    def metrics(self, action, obs, reward, done):
        raise NotImplementedError()


class RenderWrapper(gym.Wrapper):
    """Environment wrapper which automatically renders the environment at every timestep."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        super().render()
        return obs, reward, done, info


class EvalMetricCallback(EvalCallback):
    """Custom EvalCallback supporting:
        * Automatic saving of VecNormalize environment statistics when a new best episode is encountered.
        * Aggregation and logging of metrics attached to an environment (using MetricWrapper subclasses).
        * Automatic updates of adaptive schedules based on the mean aggregation value of an attached metric.
    """

    def __init__(self, eval_env, callback_on_new_best=None, callback_after_eval=None, n_eval_episodes: int = 5, eval_freq: int = 10000,
                 adaptive_schedules: Tuple["AdaptiveSchedule"] = (), log_path: str = None, best_model_save_path: str = None, deterministic: bool = True,
                 render: bool = False, verbose: int = 1, warn: bool = True):
        # Register self._new_best_callback
        if callback_on_new_best is not None:
            callback_on_new_best = CallbackList([callback_on_new_best, ConvertCallback(self._new_best_callback)])
        else:
            callback_on_new_best = ConvertCallback(self._new_best_callback)
        # Register self._after_eval_callback
        if callback_after_eval is not None:
            # User-defined callback after log dump, such that it can access logged metrics
            callback_after_eval = CallbackList([ConvertCallback(self._after_eval_callback), callback_after_eval])
        else:
            callback_after_eval = ConvertCallback(self._after_eval_callback)
        super().__init__(eval_env, callback_on_new_best, callback_after_eval, n_eval_episodes, eval_freq, log_path,
                         best_model_save_path, deterministic, render, verbose, warn)
        # Initialize metric aggregation modes and buffers
        self._metric_aggrs = {}
        self._metric_buffers = {}
        for m, aggrs in eval_env.envs[0].aggrs.items():
            self._metric_aggrs[m] = aggrs
            self._metric_buffers[m] = []
        # Register the passed adaptive schedules
        self.adaptive_schedules = {schedule.metric: schedule for schedule in adaptive_schedules}

    def _log_success_callback(self, locals_, globals_):
        # Callback on every timestep of any evaluation episode
        super()._log_success_callback(locals_, globals_)
        info = locals_["info"]
        # VecEnv: unpack
        if not isinstance(info, dict):
            info = info[0]

        # Store all metric values for this evaluation step in their corresponding buffers:
        for metric, val in info.get("metrics", {}).items():
            if val is not None and not (np.isscalar(val) and np.isnan(val)):
                self._metric_buffers[metric].append(val)

    def _new_best_callback(self, *args, **kwargs):
        """
        Save statistics of VecNormalize environment, if applicable.
        """
        eval_env = self.eval_env
        while isinstance(eval_env, VecEnvWrapper):
            if isinstance(eval_env, VecNormalize):
                eval_env.save(str(pathlib.Path(self.best_model_save_path, "normalize_stats.pkl")))
            eval_env = eval_env.venv
        return True

    def _log_metric(self, metric, data, aggrs):
        """
        Aggregate the given metric data and log or save it.
        """
        log_aggr = {
            "min": np.min,
            "max": np.max,
            "mean": np.mean,
            "std": np.std,
            "sum": np.sum
        }  # Supported aggregation modes for logging
        if not isinstance(aggrs, (list, tuple)):
            aggrs = [aggrs]
        for aggr in aggrs:
            if aggr in log_aggr:
                # Aggregate and log the metric data
                self.logger.record(f"eval/{metric}/{aggr}", log_aggr[aggr](data))
            elif aggr == "raw":
                # Save the raw metric data
                np.save(str(pathlib.Path(self.log_path).parent / f"{metric}"), data)

    def _after_eval_callback(self, *args, **kwargs):
        # Log requested aggregations of all collected metric data:
        for metric, buf in self._metric_buffers.items():
            data = np.array(buf)
            self._log_metric(metric, data, self._metric_aggrs[metric])
            if metric in self.adaptive_schedules:
                # Update any adaptive schedules using this metric
                self.adaptive_schedules[metric].adapt(np.mean(data))
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)
        # Clear metric buffers for next evaluation
        for buf in self._metric_buffers.values():
            buf.clear()
        return True


class AdaptiveSchedule(object):
    """ Adapt value based on the mean value of an evaluation metric. """

    def __init__(self, metric, incr_threshold, decr_threshold=None, invert=False, initial_value=1, min_value=1e-10, max_value=1e10, incr_factor=1.4, decr_factor=0.7):
        self.metric = metric
        if decr_threshold is None:
            decr_threshold = -incr_threshold
        self.incr_threshold = incr_threshold
        self.decr_threshold = decr_threshold
        self.invert = invert

        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.incr_factor = incr_factor
        self.decr_factor = decr_factor

    def __call__(self, progress_remaining):
        return self.value

    def adapt(self, metric):
        if metric > self.incr_threshold:
            self.value *= self.incr_factor if not self.invert else self.decr_factor
        elif metric < self.decr_threshold:
            self.value *= self.decr_factor if not self.invert else self.incr_factor
        self.value = np.clip(self.value, self.min_value, self.max_value)


def get_linear_fn(start: float, end: float, start_fraction: float = 0.0, end_fraction: float = 1.0) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = ``start_fraction`` and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params start_fraction: fraction of ``progress_remaining``
        where linear interpolation starts. E.g. 0.1 then start is
        returned for the first 10% of the training process, after
        which the interpolation towards end starts.
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.2 then end is reached after 20%
        of the complete training process.
    :return:
    """

    def func(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        if progress < start_fraction:
            return start
        elif progress > end_fraction:
            return end
        else:
            return start + (progress - start_fraction) * (end - start) / (end_fraction - start_fraction)
    func.cfg = {
        "start": start,
        "end": end,
        "start_fraction": start_fraction,
        "end_fraction": end_fraction
    }

    return func


class TileVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple existing environments, calling each environment in sequence on the
    current Python process. Only used to plot multiple existing environments in a single window.

    :param envs: a list of environments
    """

    def __init__(self, envs: List[gym.Env], names=None, rows=None, cols=None):
        self.envs = envs
        N = len(envs)
        env = self.envs[0]
        VecEnv.__init__(self, N, env.observation_space, env.action_space)
        if names is None:
            names = [f"Env_{i}" for i in range(N)]
        self.names = names
        self.img_font = ImageFont.truetype("arial.ttf", 20)
        if rows is None and cols is None:
            rows = int(np.ceil(np.sqrt(N)))
        if rows is None:
            rows = int(np.ceil(N / cols))
        if cols is None:
            cols = int(np.ceil(N / rows))
        self.rows = rows  # Number of rows to place images in a grid
        self.cols = cols  # Number of columns to place images in a grid

        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        buf_obs, buf_rews, buf_dones, buf_infos = [], [], [], []
        for env_idx in range(self.num_envs):
            obs, r, done, info = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            if done:
                obs = self.envs[env_idx].reset()
            buf_obs.append(obs)
            buf_rews.append(r)
            buf_dones.append(done)
            buf_infos.append(info)
        return (buf_obs, buf_rews, buf_dones, buf_infos)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self):
        obs_buf = []
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            obs_buf.append(obs)
        return obs_buf

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        imgs = []
        for env, name in zip(self.envs, self.names):
            img = Image.fromarray(env.render(mode="rgb_array"))
            draw = ImageDraw.Draw(img)
            x, y = int(img.width / 2), 25
            w, h = self.img_font.getsize(name)
            draw.rectangle((x - w/2, y - h/2, x + w/2, y + h/2), fill="white")
            draw.text((x, y), name, "black", font=self.img_font, anchor="mm")
            img = ImageOps.expand(img, 2)
            imgs.append(np.array(img))
        return imgs

    def _tile_images(self, imgs):
        N = len(imgs)  # Number of images
        H = self.rows  # Number of rows to place images in a grid
        W = self.cols  # Number of columns to place images in a grid
        h, w, c = imgs[0].shape  # Height, width and channel depth of each individual image
        imgs.extend([np.zeros((h, w, c), np.uint8) for _ in range(W * H - N)])  # Remaining space in grid is black
        imgs = np.array(imgs).reshape((H, W, h, w, c))  # Create large array containing all images in their respective grid location
        img = imgs.swapaxes(1, 2).reshape((H * h, W * w, c))  # And create one large image from those
        return img

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            bigimg = self._tile_images(self.get_images())
            if mode == "human":
                import cv2  # pytype:disable=import-error

                cv2.imshow("vecenv", bigimg[:, :, ::-1])
                cv2.waitKey(1)
            elif mode == "rgb_array":
                return bigimg
            else:
                raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
