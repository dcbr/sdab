"""
Implementation of the truncated Gaussian distribution for pytorch and stable baselines.

Contents of this file are based on https://github.com/toshas/torch_truncnorm whose license is included below.

BSD 3-Clause License

Copyright (c) 2020, Anton Obukhov
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import math
from numbers import Number
from typing import Tuple
import stable_baselines3.common.distributions as sb_dist

import torch
from torch import nn
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'low': constraints.real,
        'high': constraints.real,
    }
    has_rsample = True

    def __init__(self, low, high, validate_args=None):
        self.l, self.u = broadcast_all(low, high)
        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.l.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.l.dtype != self.u.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.l >= self.u).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.l.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._phi_l = self._phi(self.l)
        self._phi_u = self._phi(self.u)
        self._Phi_l = self._Phi(self.l)
        self._Phi_u = self._Phi(self.u)
        self._Z = (self._Phi_u - self._Phi_l).clamp_min(eps)
        self._log_Z = self._Z.log()
        safe_l = torch.nan_to_num(self.l, nan=math.nan)  # Deal with infinity lower/upper bounds
        safe_u = torch.nan_to_num(self.u, nan=math.nan)
        norm_u_phi_u_min_l_phi_l = (self._phi_u * safe_u - self._phi_l * safe_l) / self._Z
        self._mean = -(self._phi_u - self._phi_l) / self._Z
        self._variance = 1 - norm_u_phi_u_min_l_phi_l - ((self._phi_u - self._phi_l) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * norm_u_phi_u_min_l_phi_l

    @property
    def low(self):
        return self.l

    @property
    def high(self):
        return self.u

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.l, self.u)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _phi(x):
        """Probability density function of normal distribution."""
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _Phi(x):
        """Cumulative distribution function of normal distribution."""
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_Phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._Phi(value) - self._Phi_l) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_Phi(self._Phi_l + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.l.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, low, high, validate_args=None):
        self.loc, self.scale, self._low, self._high = broadcast_all(loc, scale, low, high)
        std_low = (self._low - self.loc) / self.scale
        std_high = (self._high - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(std_low, std_high, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std(self, value):
        return ((value - self.loc) / self.scale).clamp(self.l, self.u)

    def _from_std(self, value):
        return (value * self.scale + self.loc).clamp(self._low, self._high)

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std(value))

    def icdf(self, value):
        return self._from_std(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std(value)) - self._log_scale


class TruncGaussianDistribution(sb_dist.Distribution):  # SB Distribution class for Truncated Gaussian distributions
    """
    Truncated Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super(TruncGaussianDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> "TruncGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std, lower bound, upper bound)
        """
        self.distribution = TruncatedNormal(mean_actions, log_std.exp(), low, high)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sb_dist.sum_independent_dims(log_prob)

    def entropy(self) -> torch.Tensor:
        return sb_dist.sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, low: torch.Tensor, high: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, low, high)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std, low, high)
        log_prob = self.log_prob(actions)
        return actions, log_prob


if __name__ == "__main__":
    # Verify (implicit) gradient of truncated Gaussian samples w.r.t. distribution parameters
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    def implicit_grads(samples, mu, std, l, u):
        xi = (samples - mu) / std
        lam = (l - mu) / std
        ups = (u - mu) / std
        norm = torch.distributions.Normal(0.0, 1.0)

        N = norm.log_prob(xi).exp() * (norm.cdf(ups) - norm.cdf(lam))
        dmu = ((norm.log_prob(lam).exp() - norm.log_prob(ups).exp()) * (norm.cdf(xi) - norm.cdf(lam)) - (
                    norm.log_prob(lam).exp() - norm.log_prob(xi).exp()) * (norm.cdf(ups) - norm.cdf(lam))) / N
        dstd = ((norm.log_prob(lam).exp() * lam - norm.log_prob(ups).exp() * ups) * (norm.cdf(xi) - norm.cdf(lam)) - (
                    norm.log_prob(lam).exp() * lam - norm.log_prob(xi).exp() * xi) * (norm.cdf(ups) - norm.cdf(lam))) / N
        dl = norm.log_prob(lam).exp() * (norm.cdf(ups) - norm.cdf(xi)) / N
        du = norm.log_prob(ups).exp() * (norm.cdf(xi) - norm.cdf(lam)) / N

        return (dmu, dstd, dl, du)


    means = [-1.0, -0.95, -0.5, 0.0, 0.5, 0.95, 1.0]
    stds = [0.1, 0.5, 1.0, 2.0]
    errs = np.empty((len(means), len(stds), 4))

    for i, mu_s in enumerate(means):
        for j, std_s in enumerate(stds):
            mu = torch.tensor(mu_s, requires_grad=True)
            std = torch.tensor(std_s, requires_grad=True)
            l = torch.tensor(-1.0, requires_grad=True)
            u = torch.tensor(1.0, requires_grad=True)

            trunc_dist = TruncatedNormal(mu, std, l, u)
            samples = trunc_dist.rsample([10000])
            samples.sum().backward()
            grads = tuple(t.grad for t in (mu, std, l, u))
            with torch.no_grad():
                imp_grads = tuple(grad.sum() for grad in implicit_grads(samples, mu, std, l, u))
            errs[i,j,:] = [(grad - imp_grad).abs() for (grad, imp_grad) in zip(grads, imp_grads)]

    for i, comp in enumerate(["mu", "std", "l", "u"]):
        f, ax = plt.subplots()
        im = ax.imshow(errs[:,:,i])
        ax.set_title(f"Error on {comp} gradient")
        ax.set_xticks(np.arange(len(stds)))
        ax.set_xticklabels(stds)
        ax.set_yticks(np.arange(len(means)))
        ax.set_yticklabels(means)
        f.colorbar(im)
        plt.show()
