"""
Provides an abstract class for Distributional Causal Nodes (DCN).
"""

import torch

from ..node import CausalNode
from ..utils import requires_init
from .parameters import ParameterNet

from functools import partial


class DCN(CausalNode):
    """Abstract class for any Distributional Causal Node (DCN).

    Any class that inherits from this one must override the following methods:
    ```python
        def _params(self, dim): 
            # Returns the tuple of parameters 
            # that define this node's r.v. distribution with dimension dim.
            ...
            
        def _warm_start(self, x, *params):
            # Warm starts the node with some values x, given the parameters. 
            # Optional, default doesn't do anything.
            ...

        def _sample(self, n, *params, *ex): 
            # Sample from the distribution, given the parameters.
            ...

        def _loglk(self, x, *params):
            # Compute the loglk of x given the parameters.
            ...

        def _abduct(self, x, *params):
            # Inverts (or samples from the distribution of) ex_noise
            # given x and params.
            ...
    ```

    Also, override the class attribute:
        - discrete (bool): whether the corresponding r.v. is discrete.
    
    And the instance attributes:
        - ex_noise (tuple)
        - ex_invertible (bool)
    """

    @property
    def normalize(self):
        return self._normalize_bool.item()
    
    @property
    def requires_initialization(self):
        return self.normalize

    latent = False # DCNs are learnable, so always non-latent

    def __init__(
        self, name, *parents, net_f=None,
        normalize=True, normalize_eps=1e-6, 
        head_parents=None, **kwargs
    ):
        """
        Args:
            name (str): name of the node.
            *parents (CausalNode): nodes that act as parents for this node.
            net_f (function): 
                function(input_dim, output_dim, head_slices=None) that returns 
                a network used for parameter estimation, given parents values.
                If head_parents is not None, head_slices contains the slices
                corresponding to each head_parent.
            normalize (bool): whether to normalize the values of this node
                while executing sample and loglk for better numerical stability.
            normalize_eps (float): minimum value for scale during normalization.
        """

        super().__init__(name, *parents, **kwargs)

        # Note that we exclude ex_noise from input_dim, 
        # since they don't enter the net
        self.input_dim = sum(p.dim for p in self.parents)

        params = self._params(self.dim)
        if not any(isinstance(params, t) for t in (list, tuple)):
            params = (params,)
        self.params = tuple(params)
        
        # If there are head_parents, determine their slices
        if net_f is not None and head_parents:
            head_slices = []
            for pname in head_parents:
                start = 0
                for parent in parents:
                    end = start + parent.dim
                    if parent.name == pname:
                        head_slices.append(slice(start, end))
                        break
                    start = end

            net_f = partial(net_f, head_slices=head_slices)

        self.params_net = ParameterNet(
            self.input_dim, *self.params, net_f=net_f
        )
        
        if self.params_net.external_theta:
            self.theta_dim = self.params_net.theta_dim
            self.theta_init = self.params_net.theta_init

        self.register_buffer('_normalize_bool', torch.tensor(normalize))
        self.register_buffer('_normalize_eps', torch.tensor(normalize_eps))

        if self.normalize:
            self.register_buffer('_normalize_bias', torch.randn(self.dim))
            self.register_buffer('_normalize_log_weight', torch.randn(self.dim))


    # Normalize functions
    # Note that normalize is taken care of by warm_start next:

    def _normalize(self, x):
        """Normalize x with BatchNorm.

        Returns (x, log|det T(x)|) where T(x) is BatchNorm(x).
        """

        loc, log_scale = self._normalize_bias, self._normalize_log_weight
        scale = torch.exp(log_scale)

        x = (x - loc) / scale
        log_abs_det = -log_scale.unsqueeze(0).sum(dim=1)

        return x, log_abs_det

    def _denormalize(self, x):
        """Denormalize a normalized x from the parameters in BatchNorm.

        Returns x denormalized.
        """

        loc, log_scale = self._normalize_bias, self._normalize_log_weight
        scale = torch.exp(log_scale)

        x = x * scale + loc
        log_abs_det = log_scale.unsqueeze(0).sum(dim=1)

        return x, log_abs_det


    # Overrides for CausalNode:
    def _process_parents(self, n, *parents):
        # Override _process_parents to split parents and ex 
        # and stack each group in single tensors.
        _parents = super()._process_parents(n, *parents)
        parents, ex = \
            _parents[:-len(self.ex_noise)], _parents[-len(self.ex_noise):]

        return parents, ex
    
    def warm_start(self, x, *parents, theta=None):
        if self.normalize:
            self._normalize_bias.data = x.mean(0)
            self._normalize_log_weight.data = torch.log(
                x.std(0) + self._normalize_eps
            )
            
            x, _ = self._normalize(x)
            
        n = x.size(0)
        parents, ex = self._process_parents(n, parents)
        
        self.params_net.warm_start(x, *parents, theta=theta)
        with torch.no_grad():
            params = self.params_net(n, *parents, theta=theta)
            
        self._warm_start(x, *params)
        
        return super(CausalNode, self).warm_start(x) # skip CausalNode's warm_start

    @requires_init
    def sample(self, n, *parents, theta=None):
        parents, ex = self._process_parents(n, parents)

        params = self.params_net(n, *parents, theta=theta)
        x = self._sample(n, *params, *ex)
        
        assert x.shape == (n, self.dim), (x.shape, (n, self.dim))
        assert x.device.type == self.device.type, \
            (x.device, self.device)

        if self.normalize:
            x, _ = self._denormalize(x)

        return x

    @requires_init
    def loglk(self, x, *parents, theta=None):
        n = x.size(0)
        parents, _ = self._process_parents(n, parents)

        params = self.params_net(n, *parents, theta=theta)

        if self.normalize:
            x, log_abs_det = self._normalize(x)

        loglk = self._loglk(x, *params)
        assert loglk.shape == (n,), (self, loglk.shape, n)

        if self.normalize:
            loglk = loglk + log_abs_det

        return loglk

    @requires_init
    def abduct(self, x, *parents, theta=None):
        n = x.size(0)
        parents, _ = self._process_parents(n, parents)

        params = self.params_net(n, *parents, theta=theta)

        if self.normalize:
            x, _ = self._normalize(x)

        ex = self._abduct(x, *params)
        
        if not any(isinstance(ex, t) for t in (list, tuple, set)):
            ex = (ex,)

        assert (e.shape == (n, ex.dim) for e, ex in zip(ex, self.ex_noise))

        return tuple(ex)


    # To override:
    def _params(self, dim):
        """Create Parameter instances."""
        raise NotImplementedError()
        
    def _warm_start(self, x, *params):
        return # default, doesn't do anything

    def _sample(self, n, *args): # args = *params, *ex_noise
        """Return a sample given the r.v. params and an ex_noise sample.
    
        Args:
            n (int): number of samples to sample.
            *args (torch.Tensor): contains both *params and *ex_noise.
        """
        raise NotImplementedError()

    def _loglk(self, x, *params):
        """Return the loglk of x given the distribution's parameters.

        Args:
            x (torch.Tensor): values to compute the loglk of.
            *params (torch.Tensor): tensors with the values 
                for each parameter in self.params.
        """
        raise NotImplementedError()

    def _abduct(self, x, *params):
        """Return ex_noise values such that the node with this params creates x.

        Args:
            x (torch.Tensor): values to recreate from ex_noise.
            *params (torch.Tensor): tensors with the values
                for each parameter in self.params.
        """
        raise NotImplementedError()