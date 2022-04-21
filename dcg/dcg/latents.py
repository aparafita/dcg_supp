"""
Provides latent CausalNodes to use them as exogenous noise signals.

* `LatentNode`: abstract class for any latent nodes.
* `LatentUniform`: Uniform distribution latent node.
* `LatentNormal`: Normal distribution latent node.
* `LatentGumbel`: Gumbel distribution latent node.
"""

import numpy as np
import torch

from .node import CausalNode
from .sampling import uniform, gumbel


class LatentNode(CausalNode):
    """Abstract latent causal node.

    Used for defining exogenous noise signals or confounding latent nodes.

    Any class that inherits from LatentNode needs to override:
    ```python
        def _sample(self, n, theta=None):
            # Sample from the node's distribution. Ignore theta.
            ...

        def _loglk(self, x, theta=None): 
            # Compute the loglk of x. Ignore theta.
            ...
    ```

    Additionally, it should define the class attribute `discrete`, 
    which determines if the latent distribution is discrete.
    """

    latent = True

    def __init__(self, *args, **kwargs):
        assert kwargs.pop('latent', True), self

        super().__init__(*args, latent=True, **kwargs)
        
        self.ex_noise = []
        self.ex_invertible = True


class Uniform(LatentNode):
    """LatentNode defined by a U(a, b) distribution"""

    discrete = False

    def __init__(self, *args, a=0., b=1., eps=1e-6, **kwargs):
        super().__init__(*args, **kwargs)

        assert a < b, (self, a, b)
        self.a = a
        self.b = b

        self.eps = eps

    def _sample(self, n, theta=None):
        a, b = self.a, self.b
        
        u = uniform(n, self.dim, eps=self.eps, device=self.device)
        return u * (b - a) + a

    def _loglk(self, x, theta=None):
        return torch.zeros_like(x[:, 0], device=x.device)


class Normal(LatentNode):
    """LatentNode defined by a N(0, Id) distribution"""

    discrete = False

    def _sample(self, n, theta=None):
        return torch.randn(n, self.dim, device=self.device)

    def _loglk(self, x, theta=None):
        return -.5 * (self.dim * np.log(2 * np.pi) + (x ** 2).sum(dim=1))


class Gumbel(LatentNode):
    """LatentNode defined by a Gumbel(0, 1) distribution"""

    discrete = False

    def _sample(self, n, theta=None):
        return gumbel(n, self.dim, device=self.device)

    def _loglk(self, x, theta=None):
        return -x - torch.exp(-x)
