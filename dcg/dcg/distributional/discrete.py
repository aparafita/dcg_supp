"""Includes implementations for DCNs with discrete random variables."""

from functools import partial

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .distributional import DCN
from .parameters import *
from ..latents import Gumbel, Uniform
from ..sampling import gumbel, categorical, uniform
from ..utils import log_sum_exp_trick

from scipy import stats
from torch import distributions


class DiscreteDCN(DCN):
    """Abstract class for Discrete DCNs.

    Only ensures that all nodes are marked as discrete.
    Also that these nodes are not normalized.
    
    Remember to override methods _params, _warm_start (optional), _sample, _loglk, _abduct,
    and the instance attributes ex_noise, ex_invertible.
    Check `DCN`'s doc for more information.
    """

    discrete = True

    def __init__(self, *args, **kwargs):
        """"""
        kwargs.pop('normalize', False) # we shouldn't normalize in discrete r.v.

        super().__init__(*args, normalize=False, **kwargs)

        assert self.discrete, self


def _abduct_gumbel(x, log_p):
    """Given a one-hot x and the corresponding log_p parameters,
    samples from the abducted Gumbel distribution that produced this sample.
    """
    n = x.size(0)
    device = x.device
    
    g_k = gumbel(n, 1, device=device)
    g = -torch.log(
        torch.exp(-(gumbel(n, x.size(1), device=device) + log_p)) + 
        torch.exp(-g_k)
    )
    
    g = x * g_k + (1 - x) * g # join in a single tensor
    g = g - log_p # From Gumbel(log_p, 1) to Gumbel(0, 1)
    
    return g


class Categorical(DiscreteDCN):
    """Categorical distribution. 

    Use this class (preferably) when #categories > 2.
    Otherwise, use Bernoulli."""

    def __init__(self, name, *parents, temperature=1e-2, **kwargs):
        assert kwargs.get('dim', 1) > 1, \
            f'Categorical defined by dim > 1: {self}'

        super().__init__(name, *parents, **kwargs)
        
        self.ex_noise = Gumbel(self.name + '.ex', dim=self.dim)
        self.ex_invertible = False

        self.register_buffer('temperature', torch.tensor(temperature))


    # Overrides:
    def _params(self, dim):
        return ParameterCategorical(dim=dim)

    def _sample(self, n, log_p, ex):
        return categorical(
            n, log_p, ex, 
            soft=False, temperature=self.temperature
        )

    def _loglk(self, x, log_p):
        return (x * log_p).sum(dim=1)

    def _abduct(self, x, log_p):
        return _abduct_gumbel(x, log_p)


class Bernoulli(DiscreteDCN):
    """Bernoulli distribution.

    Even though it could be modelled as a Categorical(dim=2), 
    Bernoullis have 1 dimension. That way, we only learn 1 parameter.
    """

    def __init__(self, name, *parents, temperature=1e-2, **kwargs):
        assert kwargs.get('dim', 1) == 1, \
            f'Bernoulli defined by dim == 1: {self}'

        super().__init__(name, *parents, **kwargs)
        
        self.ex_noise = Gumbel(self.name + '.ex', dim=2)
        self.ex_invertible = False

        self.register_buffer('temperature', torch.tensor(temperature))

    # Overrides:
    def _params(self, dim):
        return ParameterBernoulli()

    def _activation(self, log_p):
        # Need to transform our 1-dimensional log_p pre-activation tensor
        # into a 2-dimensional post-activation log_p tensor.
        return self.params[0].activation(log_p)

    def _sample(self, n, log_p, ex):
        log_p = self._activation(log_p)

        return categorical(
            n, log_p, ex, 
            soft=False, temperature=self.temperature
        )[:, [1]]

    def _loglk(self, x, log_p):
        x = torch.cat([1 - x, x], 1)
        log_p = self._activation(log_p)

        return (x * log_p).sum(dim=1)

    def _abduct(self, x, log_p):
        x = torch.cat([1 - x, x], 1)
        log_p = self._activation(log_p)

        return _abduct_gumbel(x, log_p)


'''
class Poisson(DiscreteDCN):
    """Poisson distribution.
    
    Note that if dim > 1, 
    they will be mutually independent Poissons given their parents.
    """

    def __init__(self, name, *parents, eps=1e-3, **kwargs):
        super().__init__(name, *parents, **kwargs)
        
        self.ex_noise = Uniform(self.name + '.ex', dim=self.dim)
        self.ex_invertible = False

        self.register_buffer('eps', torch.tensor(eps))

    # Overrides:
    def _params(self, dim):
        return ParameterLogScale(dim=dim)

    def _sample(self, n, log_l, ex):
        # Using inverse transform sampling
        x = torch.zeros_like(log_l)
    
        if ex is None:
            ex = uniform(*log_l.shape, device=log_l.device, eps=self.eps)

        log_ex = torch.log(ex)

        log_p = -torch.exp(log_l)
        log_s = log_p.clone()

        idx = log_ex > log_s
        current_x = 0
        while idx.any().item():
            current_x += 1
            x[idx] += 1

            log_p[idx] += log_l[idx] - np.log(current_x)
            log_s[idx] = log_sum_exp_trick(
                torch.stack([log_s[idx], log_p[idx]], -1)
            )

            idx = log_ex > log_s

        return x

    def _loglk(self, x, log_l):
        x = x.float().round().long()
        
        i = torch.arange(x.max().item() + 1, device=x.device).float()
        
        i[0] = 1
        log_fact = torch.cumsum(torch.log(i), 0)
        
        return (x * log_l - torch.exp(log_l) - log_fact[x]).sum(dim=1)

    def _abduct(self, x, log_l):
        x = x.float().round().int().clone()
        ex = uniform(*log_l.shape, device=x.device)

        log_p = -torch.exp(log_l)
        log_s = log_p.clone()

        # Update u for those x == 0
        idx = x == 0
        ex[idx] *= torch.exp(log_s[idx])

        current_x = 0
        while (x > 0).any().item():
            current_x += 1
            x -= 1
            idx = x >= 0
            idx2 = x == 0

            log_p[idx] += log_l[idx] - np.log(current_x)

            prev_s = torch.exp(log_s[idx2])
            log_s[idx] = log_sum_exp_trick(
                torch.stack([log_s[idx], log_p[idx]], -1)
            )

            # Update u for those x == 0
            ex[idx2] = ex[idx2] * (torch.exp(log_s[idx2]) - prev_s) + prev_s

        ex = ex.clamp(self.eps / 2, 1 - self.eps / 2)

        return ex
'''

class Poisson(DiscreteDCN):
    """Poisson distribution.
    
    Note that if dim > 1, 
    they will be mutually independent Poissons given their parents.
    """

    def __init__(self, name, *parents, eps=1e-3, **kwargs):
        super().__init__(name, *parents, **kwargs)
        
        self.ex_noise = Uniform(self.name + '.ex', dim=self.dim)
        self.ex_invertible = False

        self.register_buffer('eps', torch.tensor(eps))

    # Overrides:
    def _params(self, dim):
        return ParameterScale(dim=dim)
    
    def _get_scipy(self, l):
        l = l.cpu().detach().numpy()

        return stats.poisson(l)

    def _sample(self, n, l, ex):
        distr = self._get_scipy(l)
        sample = distr.ppf(ex.cpu().detach().numpy())
        
        sample = torch.Tensor(sample.astype(float)).to(l.device)
            
        return sample

    def _loglk(self, x, l):        
        return distributions.poisson.Poisson(l).log_prob(x).sum(1)

    def _abduct(self, x, l):
        distr = self._get_scipy(l)
        ex = distr.cdf(x.cpu().detach().numpy())
        
        ex = torch.Tensor(ex).to(l.device)
            
        return ex