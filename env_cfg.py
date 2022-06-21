import numpy as np
from sb_utils import get_linear_fn, AdaptiveSchedule

# Tuned hyperparameters for SAC from stable-baselines ZOO, with some minor modifications

env_cfg = {
    "LunarLanderContinuous-v2": {
        "sac": {
            "total_timesteps": int(5e5),
            "norm_kwargs": {
                "norm_obs": False,
                "norm_reward": True
            },
            "policy": 'MlpPolicy',
            "kwargs": {
                "gamma": 0.99,
                "buffer_size": int(1e6),
                "batch_size": 256,
                "learning_starts": 10000,
                "ent_coef": "auto",
                "gradient_steps": -1,  # 1
                "train_freq": (1, "episode"),  # 1
                "learning_rate": 7e-4,  # "lin_7.3e-4"
                "policy_kwargs": dict(net_arch=[400, 300])
            }
        },
    },
    "Pendulum-v1": {
        "sac": {
            "total_timesteps": 50000,
            "norm_kwargs": {
                "norm_obs": False,
                "norm_reward": True
            },
            "policy": 'MlpPolicy',
            "kwargs": {
                "gamma": 0.98,
                "buffer_size": 200000,
                "learning_starts": 10000,
                "ent_coef": "auto",
                "gradient_steps": -1,
                "train_freq": (1, "episode"),
                "learning_rate": 1e-3,
                "policy_kwargs": dict(net_arch=[400, 300])
            }
        },
    },
}


def add_to_cfg(envs, algs, key):
    if not isinstance(envs, (list, tuple)):
        envs = (envs,)
    if not isinstance(algs, (list, tuple)):
        algs = (algs,)
    if not isinstance(key, (list, tuple)):
        key = (key,)

    def decorator(f):
        for env in envs:
            for alg in algs:
                obj = env_cfg[env][alg]
                for k in key[:-1]:
                    obj = obj.setdefault(k, {})
                obj[key[-1]] = f
        return f
    return decorator


# Note on bound functions: Input observations have shape BxO with B the batch size and O the shape of an individual
# observation. The output bounds have shape 2xBxA with A the shape of an individual action.
@add_to_cfg("LunarLanderContinuous-v2", "sac", ["bounds", "stabilize"])
def llc_bounds(obs):
    # Note: these bound calculations assume no normalization of observations takes place (using VecNormalize wrapper)!
    VEL_TH = 0.2
    ANG_TH = 0.2
    ANG_VEL_TH = 0.12
    EPS = 0.01  # 0
    vel = obs[:,3]
    ang = obs[:,4]
    ang_vel = obs[:,5]
    lower = -np.ones((ang.shape[0], 2))
    upper = np.ones((ang.shape[0], 2))
    # Boolean state masks
    Tleft, Tright = ang > ANG_TH, ang < -ANG_TH  # Tilting too much to the left or right
    Rleft, Rright = ang_vel > ANG_VEL_TH, ang_vel < -ANG_VEL_TH  # Rotating too much to the left or right
    # Soft landing
    lower[vel < -VEL_TH, 0] = 0 + EPS  # Forcefully enable main engine if we are descending too fast
    # Stabilize lander
    lower[Tleft & ~Rright, 1] = 0.5 + EPS  # Forcefully enable left engine if we are tilting too much to the left
    upper[Tright & ~Rleft, 1] = -0.5 - EPS  # Forcefully enable right engine if we are tilting too much to the right
    return [lower, upper]


@add_to_cfg("Pendulum-v1", "sac", ["bounds", "stabilize"])
def pendulum_bounds_stabilize(obs):
    """Bounds for the Pendulum-v1 environment, trying to aid in the stabilization of the pendulum around the upward
    position."""
    # Note: these bound calculations assume no normalization of observations takes place (using VecNormalize wrapper)!
    ANG_TH = 0.3
    VEL_TH = 0.3
    TORQUE_EPS = 0.1
    upward = obs[:,0] > 0
    ang = obs[:,1]  # This is actually sin(theta), but we only consider small angles around 0 => sin(x) ~= x
    vel = obs[:,2]
    lower = -2*np.ones((vel.shape[0], 1))
    upper = 2*np.ones((vel.shape[0], 1))
    # Slow down pendulum when approaching the upward position at high velocity
    Vpos = upward & (vel < -VEL_TH) & (np.abs(ang) < ANG_TH)
    Vneg = upward & (vel > VEL_TH) & (np.abs(ang) < ANG_TH)
    lower[Vpos, 0] = 2-TORQUE_EPS
    upper[Vneg, 0] = -2+TORQUE_EPS
    # Prevent tipping over the pendulum around the upward position
    Spos = upward & (np.abs(vel) <= VEL_TH) & (-ANG_TH < ang <= 0)
    Sneg = upward & (np.abs(vel) <= VEL_TH) & (0 <= ang < ANG_TH)
    lower[Spos, 0] = -(2-TORQUE_EPS)*ang[Spos]/ANG_TH
    upper[Sneg, 0] = (-2+TORQUE_EPS)*ang[Sneg]/ANG_TH
    return [lower, upper]


@add_to_cfg("Pendulum-v1", "sac", ["bounds", "avoid"])
def pendulum_bounds_avoid(obs):
    """Bounds for the Pendulum-v1 environment, trying to avoid the rightmost area."""
    # Note: these bound calculations assume no normalization of observations takes place (using VecNormalize wrapper)!
    ANG_TH = 0.3
    VEL_TH = 6.0
    TORQUE_EPS = 0.1
    upward = obs[:,0] > 0
    ang = obs[:,1]  # This is actually sin(theta), but we only consider small angles around 0 => sin(x) ~= x
    vel = obs[:,2]
    lower = -2*np.ones((vel.shape[0], 1))
    upper = 2*np.ones((vel.shape[0], 1))
    # Slow down pendulum when approaching the right area at high velocity
    Vpos = upward & (vel < 0) & (ang < ANG_TH)
    Vneg = ~upward & (vel > 0) & (ang < ANG_TH)
    lower[Vpos, 0] = (2-TORQUE_EPS)*(2*np.minimum(-vel[Vpos], VEL_TH)/VEL_TH - 1)
    upper[Vneg, 0] = (-2+TORQUE_EPS)*(2*np.minimum(vel[Vneg], VEL_TH)/VEL_TH - 1)
    # Force pendulum away from right area
    Spos = upward & (ang < ANG_TH)
    Sneg = ~upward & (ang < ANG_TH)
    lower[Spos, 0] = np.maximum(lower[Spos], (2-TORQUE_EPS)*np.clip(-ang[Spos], -ANG_TH, ANG_TH)/ANG_TH)
    upper[Sneg, 0] = np.minimum(upper[Sneg], (-2+TORQUE_EPS)*np.clip(-ang[Sneg], -ANG_TH, ANG_TH)/ANG_TH)
    return [lower, upper]
