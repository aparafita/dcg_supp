"""Includes implementations for DCNs with continuous random variables."""

from functools import partial

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .distributional import DCN
from .parameters import * # parameters_net, Parameter, ParameterLoc, etc.

from .. import latents

from torch import distributions
from scipy import stats


class ContinuousDCN(DCN):
    """Abstract class for Continuous DCNs.

    Only ensures that all nodes are marked as non-discrete.
    
    Remember to override methods _params, _warm_start (optional), _sample, _loglk, _abduct,
    and the instance attributes ex_noise, ex_invertible.
    Check `DCN`'s doc for more information.
    """

    discrete = False

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        assert not self.discrete, self


class Normal(ContinuousDCN):
    """Normal distribution."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ex_noise = latents.Normal(self.name + '.ex', dim=self.dim)
        self.ex_invertible = True
        
    # Overrides:
    def _params(self, dim):
        return ParameterLoc(dim=dim), ParameterScale(dim=dim)

    def _sample(self, n, loc, scale, ex):
        return ex * scale + loc

    def _loglk(self, x, loc, scale):
        return -.5 * (
            self.dim * np.log(2 * np.pi) + 
            2 * torch.log(scale).sum(1) + 
            (((x - loc) / scale) ** 2).sum(1)
        )

    def _abduct(self, x, loc, scale):
        return (x - loc) / scale
    
    
class Exponential(ContinuousDCN):
    """Exponential distribution."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ex_noise = latents.Uniform(self.name + '.ex', dim=self.dim)
        self.ex_invertible = True
        
    # Overrides:
    def _params(self, dim):
        return ParameterScale(name='rate', dim=dim)

    def _sample(self, n, rate, ex):
        return -torch.log(ex) / rate

    def _loglk(self, x, rate):
        return (torch.log(rate) - rate * x).sum(1)

    def _abduct(self, x, rate):
        return torch.exp(-rate * x)


class ALD(ContinuousDCN):
    """Asymmetric Laplace Distribution."""

    def __init__(self, *args, eps=1e-3, **kwargs):
        if kwargs.get('dim', 1) > 1:
            raise NotImplementedError()

        super().__init__(*args, **kwargs)
        
        self.ex_noise = latents.Uniform(
            self.name + '.ex', dim=self.dim, eps=eps
        )
        self.ex_invertible = True


    # Overrides:
    def _params(self, dim):
        return ParameterLoc(), ParameterScale(), ParameterShape()

    def _sample(self, n, loc, scale, shape, ex):
        ex = ex * (1 / shape + shape) - shape # U(-shape, 1 / shape)

        sign = torch.sign(ex)
        sign[sign == 0] = 1.

        return loc - 1 / (scale * sign * (shape ** sign)) * torch.log(
            1 - ex * sign * (shape ** sign)
        )

    def _loglk(self, x, loc, scale, shape):
        sign = torch.sign(x - loc)
        sign[sign == 0] = 1.

        return (
            torch.log(scale) - torch.log(shape + 1 / shape) - 
            (x - loc) * scale * sign * (shape ** sign)
        ).squeeze(1)

    def _abduct(self, x, loc, scale, shape):
        sign = torch.sign(x - loc)
        sign[sign == 0] = -1.

        noise = (
            1 - torch.exp((loc - x) * scale * sign * shape ** sign)
        ) * sign * shape ** -sign
        
        return (noise + shape) / (1 / shape + shape)
    

class Beta(ContinuousDCN):
    """Beta(alpha, beta) distribution.
    
    Includes functionality to define the distribution in a (a, b) interval.
    """
    
    def __init__(self, *args, a=0., b=1., eps=1e-3, **kwargs):
        kwargs.pop('normalize', None) # remove normalize
        object.__setattr__(self, 'eps', eps) # save it before super for _params()
        
        super().__init__(*args, **kwargs, normalize=False)
        
        self.ex_noise = latents.Uniform(self.name + '.ex', dim=self.dim)
        self.ex_invertible = True
        
        self.register_buffer('a', torch.tensor(a))
        self.register_buffer('b', torch.tensor(b))
        
        del self.eps
        self.register_buffer('eps', torch.tensor(eps))
        
    # Overrides:
    def _params(self, dim):
        return ParameterScale(name='alpha', eps=self.eps, dim=dim), \
            ParameterScale(name='beta', eps=self.eps, dim=dim)
    
    def _get_scipy(self, alpha, beta):
        alpha = alpha.cpu().detach().numpy()
        beta = beta.cpu().detach().numpy()

        return stats.beta(a=alpha, b=beta)

    def _sample(self, n, alpha, beta, ex):
        distr = self._get_scipy(alpha, beta)
        sample = distr.ppf(ex.cpu().detach().numpy())
        
        sample = torch.Tensor(sample).to(alpha.device)
        sample = sample * (self.b - self.a) + self.a
            
        return sample

    def _loglk(self, x, alpha, beta):
        x = (x - self.a) / (self.b - self.a)
        x = x.clip(self.eps / 2, 1 - self.eps / 2) # avoid +-inf on extremes
        
        return distributions.beta.Beta(alpha, beta).log_prob(x).sum(1)

    def _abduct(self, x, alpha, beta):
        x = (x - self.a) / (self.b - self.a)
        
        distr = self._get_scipy(alpha, beta)
        ex = distr.cdf(x.cpu().detach().numpy())
        
        ex = torch.Tensor(ex).to(alpha.device)
            
        return ex
    

# Example of a Truncated distribution with Inverse Transform Sampling
class TruncatedNormal(ContinuousDCN):
    """TruncatedNormal distribution.
    Implemented as an example of a continuous DCN defined through Inverse Transform Sampling
    and applying truncation. This implementation could be optimized."""
    
    def __init__(self, *args, a=-np.inf, b=np.inf, **kwargs):
        kwargs.pop('normalize', None) # don't allow it
        super().__init__(*args, **kwargs, normalize=False)
        
        self.ex_noise = latents.Uniform(self.name + '.ex', dim=self.dim)
        self.ex_invertible = True
        
        assert a < b
        self.register_buffer('a', torch.tensor(a))
        self.register_buffer('b', torch.tensor(b))
        
    @property
    def inf_a(self):
        return torch.isinf(self.a)
    
    @property
    def inf_b(self):
        return torch.isinf(self.b)
    
    def _cdf_extremes(self, distr):
        cdf_a = 0. if self.inf_a else distr.cdf(self.a)
        cdf_b = 1. if self.inf_b else distr.cdf(self.b)
        
        return cdf_a, cdf_b
        
    # To override
    def _params(self, dim):
        return ParameterLoc(dim=dim), ParameterScale(dim=dim)
    
    # Overrides:
    def _distr(self, loc, scale):
        return distributions.normal.Normal(loc, scale)

    def _sample(self, n, loc, scale, ex):
        distr = self._distr(loc, scale)
        cdf_a, cdf_b = self._cdf_extremes(distr)
        
        ex = ex * (cdf_b - cdf_a) + cdf_a # move to (cdf_a, cdf_b)
        sample = distr.icdf(ex)
            
        return sample

    def _loglk(self, x, loc, scale):
        distr = self._distr(loc, scale)
        cdf_a, cdf_b = self._cdf_extremes(distr)
        
        return (distr.log_prob(x) - torch.log(cdf_b - cdf_a)).sum(dim=1)

    def _abduct(self, x, loc, scale):
        distr = self._distr(loc, scale)
        cdf_a, cdf_b = self._cdf_extremes(distr)
        
        ex = distr.cdf(x)
        ex = (ex - cdf_a) / (cdf_b - cdf_a) # back to (0, 1)
            
        return ex