import gym
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import argparse
import typing
from copy import deepcopy

from stable_baselines3.common.callbacks import ConvertCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

from sac import SAC, SACPolicy
from bsac import BSAC, BSACPolicy
from env_cfg import env_cfg
from sb_utils import MetricWrapper, RenderWrapper, EvalMetricCallback, create_noise
import utils
import plotting

# Parameter defaults/allowed values:
log_dir = "runs"
envs = ["Pendulum-v1", "LunarLanderContinuous-v2"]
bounds = ["stabilize", "avoid"]
algs = ["sac", "bsac"]
rescale = ["lin", "pwl", "hyp", "clip"]
seeds = [2345185777,  491764555, 1135283524, 2825301519, 4230406099,
         1419159101, 2412356523, 3939077467, 4136346909, 2128146857]
modes = ["train", "eval", "analyze"]
eval_seed = 3687851522
eval_episodes = 10
eval_steps = 1000
procs = 4
# All seed values above are randomly generated using:
# >>> import numpy as np
# >>> rng = np.random.default_rng()
# >>> rng.integers(2**32, size=11)

RENDER = False  # Debug flag to enable rendering of the environment while training


class ObsMetricWrapper(MetricWrapper):
    AGGREGATIONS = {"obs": "raw"}

    def metrics(self, action, obs, reward, done):
        metrics = {"obs": obs}
        return metrics


class LunarMetricWrapper(MetricWrapper):
    AGGREGATIONS = {"crash": "sum", "away": "sum"}

    def metrics(self, action, obs, reward, done):
        metrics = {}
        if self.unwrapped.game_over:
            metrics["crash"] = 1
        if abs(obs[0]) >= 1.0:
            metrics["away"] = 1
        return metrics


class Run(object):

    def __init__(self, alg_id, env_id, seed, rescaling, bound, log_dir):
        self.alg_id = alg_id
        self.env_id = env_id
        self.seed = seed
        self.rescaling = rescaling
        self.bound = bound
        self.name = alg_id if alg_id == "sac" else f"{alg_id}_{rescaling}"
        self.root_dir = log_dir
        self.log_dir = pathlib.Path(log_dir, f"{self.env_id}_{bound}", self.name, f"seed_{self.seed}")
        self.alg = {"sac": SAC, "bsac": BSAC}[alg_id]
        self.model_path = self.log_dir / "model.zip"
        self.stats_path = self.log_dir / "normalize_stats.pkl"
        self.log_name = "progress.csv"
        self.video_path = self.log_dir / "videos"
        self.eval_path = self.log_dir / "eval"

        # Use tuned hyperparameters from stable-baselines ZOO:
        self.cfg = deepcopy(env_cfg[env_id]["sac"])
        if "action_noise" in self.cfg["kwargs"]:
            env = self.create_base_env()  # Create dummy environment to inspect action space
            self.cfg["kwargs"]["action_noise"] = create_noise(self.cfg["kwargs"]["action_noise"], env.action_space.shape)
        if alg_id == "bsac":
            self.cfg["kwargs"]["policy_kwargs"]["rescale"] = rescaling
            self.cfg["kwargs"]["policy_kwargs"]["bounds"] = self.cfg["bounds"][bound]
        self.cfg["policy"] = {"sac": SACPolicy, "bsac": BSACPolicy}[alg_id]

        self._model = None
        self._env = None

    @property
    def exists(self):
        return self.model_path.exists()

    def create_base_env(self, seed=None, render=False):
        env = Monitor(gym.make(self.env_id))
        if self.env_id == "Pendulum-v1":
            env = ObsMetricWrapper(env)
        elif self.env_id == "LunarLanderContinuous-v2":
            env = LunarMetricWrapper(env)
        if render:
            env = RenderWrapper(env)
        env.seed(self.seed if seed is None else seed)
        return env

    def create_vec_env(self, seed=None, render=False):
        return DummyVecEnv([lambda: self.create_base_env(seed, render)])

    def train(self):
        def on_eval(*args, **kwargs):
            self.save()
            eval_run = Run(self.alg_id, self.env_id, self.seed, self.rescaling, self.bound, self.root_dir)
            eval_run.eval(300, False, f"eval{self._model.num_timesteps}")
            self.analyze(f"eval{self._model.num_timesteps}")
            return True

        if not self.exists:
            # Automatically normalize the input features and reward
            self._env = VecNormalize(self.create_vec_env(render=RENDER), **self.cfg["norm_kwargs"])
            eval_env = VecNormalize(self.create_vec_env(eval_seed), **self.cfg["norm_kwargs"])
            # SB3 v1.5 bug workaround:
            self._env.obs_rms = eval_env.obs_rms = None
            eval_cb = EvalMetricCallback(eval_env, callback_after_eval=ConvertCallback(on_eval), n_eval_episodes=eval_episodes,
                                         eval_freq=5000, log_path=str(self.eval_path), best_model_save_path=str(self.eval_path))

            self._model = self.alg(self.cfg["policy"], self._env, tensorboard_log=str(self.log_dir), verbose=1,
                                   seed=self.seed, **self.cfg["kwargs"])
            self._model.learn(total_timesteps=self.cfg["total_timesteps"], callback=eval_cb)
            self.save()

    def save(self):
        # Don't forget to save the VecNormalize statistics when saving the agent
        self._model.save(str(self.model_path))
        self._env.save(str(self.stats_path))

    def load(self):
        if self.exists:
            env = self.create_vec_env(eval_seed+1)
            # Uncomment to use best performing model instead of the last
            # self.model_path = self.eval_path / "best_model"
            # self.stats_path = self.eval_path / "normalize_stats.pkl"

            # Load the saved statistics
            self._env = VecNormalize.load(self.stats_path, env)
            # but do not update them at test time
            self._env.training = False

            # Load the agent
            self._model = self.alg.load(self.model_path, env=self._env)

            # Load the log
            log = pd.read_csv(str(self.log_dir / self.log_name), index_col="time/total_timesteps")

            return self._env, self._model, log
        else:
            raise RuntimeError(f"Run with parameters alg_id={self.alg_id}, env_id={self.env_id}, seed={self.seed} does"
                               f"not exist yet. Call train first, before loading.")

    def eval(self, length, video, prefix="eval"):
        env, model, log = self.load()
        # Record the video starting at the first step
        if video:
            env = VecVideoRecorder(env, str(self.video_path), record_video_trigger=lambda x: x == 0,
                                   video_length=length, name_prefix=prefix)

        action_data = np.empty((6, length) + env.action_space.shape)
        obs = env.reset()
        for i in range(length):
            actions = model.predict(obs)[0]
            action_data[0, i, :] = actions[0, :]
            action_data[1:3, i, :] = model.policy._last_bounds if self.alg_id == "bsac" else [env.action_space.low, env.action_space.high]
            action_data[3, i, :] = model.policy._last_norm_action if self.alg_id == "bsac" else actions[0, :]
            action_data[4, i, :] = model.policy._last_mean
            action_data[5, i, :] = model.policy._last_std
            obs, _, _, _ = env.step(actions)
        # Save the video
        env.close()

        # Save action plots
        plot_actions(action_data[:3,:,:], ["a", "l", "u"], save_path=self.eval_path / f"{prefix}-actions.svg")
        plot_action_dists(action_data[4,:,:], action_data[5,:,:], save_path=self.eval_path / f"{prefix}-action_dists.svg")

    def fetch_analytics_data(self, uid):
        log = pd.read_csv(str(self.log_dir / self.log_name), index_col="time/total_timesteps")
        r = log["eval/mean_reward"].rename(f"r_{uid}").dropna()
        if self.env_id == "Pendulum-v1":
            obs = np.load(str(self.eval_path / "obs.npy"))
            return r, obs, eval_episodes
        elif self.env_id == "LunarLanderContinuous-v2":
            c = log["eval/crash/sum"].rename(f"c_{uid}").dropna()
            a = log["eval/away/sum"].rename(f"a_{uid}").dropna()
            return r, c, a, eval_episodes
        # Default: just return rewards
        return (r,)

    def analyze(self, prefix):
        data = self.fetch_analytics_data(prefix)
        data = [*zip(data)]
        self.do_analysis(data, self.env_id, self.eval_path, prefix)

    @staticmethod
    def do_analysis(data, env_id, save_path, prefix=""):
        if env_id == "Pendulum-v1":
            pendulum_plots(data[1], data[2], save_path, prefix)
        elif env_id == "LunarLanderContinuous-v2":
            lunar_plots(data[1], data[2], data[3], save_path, prefix)


def plot_metrics(metrics, legends, xlabel='', ylabel='', title='', save_path=None, close=True):
    xs = [metric.index.values for metric in metrics]
    ys = [np.reshape(metric["mean"].values, (-1, 1)) for metric in metrics]
    yms = [np.reshape(metric["min"].values, (-1, 1)) for metric in metrics]
    yMs = [np.reshape(metric["max"].values, (-1, 1)) for metric in metrics]
    alpha = 0.2
    return plotting.shaded_line_plot(xs, ys, yms, yMs, legends, alpha, xlabel, ylabel, title, save_path, close)


def plot_actions(actions, legends, title=None, save_path=None, close=True):
    max_comp = 4
    y_labels = [f"Action {comp}" for comp in range(min(max_comp, actions.shape[-1]))]
    return plotting.line_plots(actions[:,:,:max_comp], legends, "timesteps", y_labels, title, save_path, close)


def plot_action_dists(means, stds, save_path=None, close=True):
    return plotting.shaded_line_plot([np.arange(means.shape[0])], [means], [means-stds], [means+stds], save_path=save_path, close=close)


def pendulum_plots(obs, nb_evals, save_path, prefix=""):
    if not (prefix == "" or prefix.endswith("-")):
        prefix = prefix + "-"
    obs = np.concatenate(obs, axis=0)
    nb_evals = int(np.sum(nb_evals))

    max_vel = 8
    ang = np.arctan2(obs[:, 1], obs[:, 0])
    vel = np.minimum(np.abs(obs[:, 2]), max_vel)  # Bound maximum velocity
    H, ang_edges, vel_edges = np.histogram2d(ang, vel, bins=[20, int(np.ceil(max_vel))],
                                             range=[[-np.pi, np.pi], [0, max_vel]])
    A, V = np.meshgrid(ang_edges, vel_edges)
    Hlog = np.log(H + 0.1)  # Use log to be able to see initial visited states prior to stabilization
    f, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.pcolormesh(A, V, Hlog.T)
    plotting._handle_figure(f, save_path / f"{prefix}obs_hist.svg", True)
    f, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_ylim([0, max_vel])
    for i in range(nb_evals):
        ax.plot(ang[200 * i:200 * (i + 1)], vel[200 * i:200 * (i + 1)])
    plotting._handle_figure(f, save_path / f"{prefix}obs.svg", True)


def lunar_plots(crashes, aways, nb_evals, save_path, prefix):
    if not (prefix == "" or prefix.endswith("-")):
        prefix = prefix + "-"
    crashes = sum(crashes)
    aways = sum(aways)
    nb_evals = int(sum(nb_evals))

    if len(crashes) > 1:
        ys = [100 * crashes / nb_evals, 100 * aways / nb_evals]
        f = plotting.stacked_fill_plot(crashes.index, ys, ["Crash", "Astray"], "steps", "Episode end [%]", close=False)
        f.axes[0].set_xlim([np.min(crashes.index), np.max(crashes.index)])
        f.axes[0].set_ylim([0, 100])
        plotting._handle_figure(f, save_path=save_path / f"{prefix}endings.svg", close=True)  # f"{prefix}endings.svg"


def alg_generator(algs, rescalings):
    for alg_id in algs:
        if alg_id == "bsac":
            for rescaling in rescalings:
                yield alg_id, rescaling
        else:
            yield alg_id, None


def env_generator(envs, bounds):
    for env_id in envs:
        for bound in bounds:
            if bound in env_cfg[env_id]["sac"]["bounds"]:
                yield env_id, bound


def run(mode, alg_id, env_id, seed, rescaling, bound, log_dir, eval_length, video):
    import pybullet_envs
    load_torch()

    if not isinstance(alg_id, typing.List):
        run = Run(alg_id, env_id, seed, rescaling, bound, log_dir)
        if mode == "train":
            run.train()
        else:
            run.eval(eval_length, video)
    else:
        algs = alg_id
        rescalings = rescaling
        seeds = seed
        if mode == "analyze":
            # Bulk analytics mode
            returns = {}
            for alg_id, rescaling in alg_generator(algs, rescalings):
                data = []
                name = ""
                for seed in seeds:
                    run = Run(alg_id, env_id, seed, rescaling, bound, log_dir)
                    name = run.name
                    # Gather analytics data:
                    data.append(run.fetch_analytics_data(seed))
                data = [*zip(*data)]
                returns[name] = utils.process_metrics(data[0])
                Run.do_analysis(data, env_id, log_dir / f"{env_id}_{bound}" / "charts", name)

            # Create figures from return statistics:
            plot_metrics(returns.values(), returns.keys(), xlabel="steps", ylabel="R", title="Average return",
                         save_path=log_dir / f"{env_id}_{bound}" / "charts" / "return.svg")


def load_torch():
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Numpy import errors on some architectures without this
    import torch
    torch.set_num_threads(1)  # Multiprocessing goes crazy slow without this


def parse_args(args=None):
    parser = argparse.ArgumentParser("State-dependent action bounds experiments")
    # General args:
    parser.add_argument('--procs', default=procs, type=int, metavar='P', help="Number of processes to use for the experiments (default: %(default)d)")
    parser.add_argument('--logdir', default=log_dir, type=str, metavar='PATH', help="Path to log directory, where experiment results and videos are stored (default: %(default)s)")
    parser.add_argument('--mode', default="train", type=str, metavar='MODE', help="Run mode: train, eval or analyze (default: %(default)s)")
    # Run args:
    parser.add_argument('--envs', nargs='*', default=envs, choices=envs + ["all"], help="Environments to experiment on (default: 'all')")
    parser.add_argument('--bounds', nargs='*', default=bounds, choices=bounds + ["all"], help="Action bounds to use in the chosen environment: stabilize or avoid (default: 'all')")
    parser.add_argument('--algs', nargs='*', default=algs, choices=algs + ["all"], help="Algorithms to train with (default: 'all')")
    parser.add_argument('--rescale', nargs='*', default=rescale, choices=rescale + ["all"], help="Rescaling (or clipping) function to use: lin, pwl, hyp or clip (default: 'all')")
    parser.add_argument('--seeds', nargs='*', default=None, type=int, metavar='S', help="Seeds to use, -N to select N random seeds (default: 10 seeds used in paper)")
    # Evaluation args:
    parser.add_argument('--evalsteps', default=eval_steps, type=int, metavar='S', help="Amount of simulated timesteps during evaluation (default: %(default)d)")
    parser.add_argument('--novideo', action="store_false", dest="video", help="Do not create videos of the evaluations")

    params = parser.parse_args(args)
    if "all" in params.envs:
        params.envs = envs
    if "all" in params.bounds:
        params.bounds = bounds
    if "all" in params.algs:
        params.algs = algs
    if "all" in params.rescale:
        params.rescale = rescale
    if params.seeds is None:
        params.seeds = seeds
    elif len(params.seeds) == 1 and params.seeds[0] < 0:
        N = -params.seeds[0]
        rng = np.random.default_rng()
        params.seeds = rng.integers(2**32, size=N).tolist()
    params.logdir = pathlib.Path(params.logdir)

    return params


if __name__ == "__main__":
    def run_args_gen(params):
        if params.mode == "analyze":
            for env_id, bound in env_generator(params.envs, params.bounds):
                yield params.mode, params.algs, env_id, params.seeds, params.rescale, bound, params.logdir, params.evalsteps, params.video
        else:
            for alg_id, rescaling in alg_generator(params.algs, params.rescale):
                for env_id, bound in env_generator(params.envs, params.bounds):
                    for seed in params.seeds:
                        yield params.mode, alg_id, env_id, seed, rescaling, bound, params.logdir, params.evalsteps, params.video

    params = parse_args()
    args_gen = run_args_gen(params)
    if params.procs > 1:
        load_torch()
        mp.set_start_method('spawn')  # This will make sure the different workers can use different seeds
        with mp.Pool(params.procs) as pool:
            pool.starmap(run, args_gen)
    else:
        for args in args_gen:
            run(*args)
