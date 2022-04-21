"""Includes an abstract class for a DCN Parameter, 
and several implementations for each type of Parameter.
"""

import numpy as np

import torch
from torch import nn

import torch.nn.functional as F

from ..utils import Module, softplus_inv


def parameters_net(
    input_dim, output_dim, n_layers=3, hidden_dim=100, use_dropout=True
):
    """Create a network for computation of parameters.

    Args:
        input_dim (int): input dimension for network.
        output_dim (int): output dimension for network. 
            Basically, concatenation of parameter dimensions.
        n_layers (int): number of layers in network.
        hidden_dim (int): dimension of hidden layers.
    """
    assert input_dim > 0, 'Net with no inputs'
    assert n_layers >= 1, 'Net with no layers'

    def step(n, k):
        if k == 0:
            return nn.Linear(
                input_dim if n == 0 else hidden_dim, 
                hidden_dim
            )
        elif k == 1:
            if use_dropout:
                return nn.Dropout()
            else:
                return None
        else:
            return nn.ReLU()

    return nn.Sequential(
        nn.BatchNorm1d(input_dim, affine=False),
        *(
            layer
            for layer in (
                step(n_layer, n_step)

                for n_layer in range(n_layers - 1)
                for n_step in range(3)
            )
            if layer is not None
        ),
        nn.Linear(input_dim if n_layers == 1 else hidden_dim, output_dim)
    )


class ParameterNet(Module):
    """Network that computes the value of a node's distribution parameters.

    Given the conditioning parent's values of a node, 
    outputs the resulting parameters for the node's assumed distribution.
    """

    def __init__(self, input_dim, *params, net_f=None):
        """
        Args:
            input_dim (int): input dimension.
            *params (Parameter): required parameters.
            net_f (function): 
                function(input_dim, output_dim) that returns 
                a network that computes parameters values.
        """

        super().__init__()

        assert params
        self.params = params

        assert input_dim >= 0
        self.input_dim = input_dim
        self.output_dim = sum(p.dim for p in params)

        self.theta_init = torch.cat([
            p if p is not None else torch.randn(dim)
            for p, dim in [ (p.init(), p.dim) for p in params ]
        ])
        assert self.theta_init.shape == (self.output_dim,)
        
        if not self.input_dim:
            self._params = nn.Parameter(self.theta_init.unsqueeze(0))
            self.theta_dim = 0
        elif net_f is not None:
            self.net = net_f(self.input_dim, self.output_dim, init=self.theta_init)
            self.theta_dim = 0
        else:
            self.theta_dim = self.output_dim
            
        self.external_theta = bool(self.theta_dim)
        self.requires_initialization = (
            hasattr(self, 'net') and 
            getattr(self.net, 'requires_initialization', False)
        )
                
            
    def _update_device(self, device):
        super()._update_device(device)
        
        for param in self.params:
            param.device = device

    def forward(self, n, *parents, theta=None):
        assert all(p.size(0) == n for p in parents) and (
            not parents or sum(p.size(1) for p in parents) == self.input_dim
        )
        
        assert (self.theta_dim and theta is not None) or \
               (not self.theta_dim and theta is None)
        
        if self.external_theta:
            assert theta is not None
            params = theta
        else:
            assert theta is None

            # Prepare input tensor
            if parents:
                params = self.net(torch.cat(parents, dim=1))
            else:
                params = self._params.repeat(n, 1)

        assert params.shape == (n, self.output_dim), \
            f"Parameter shape {params.shape} doesn't match "\
            f"expected shape: (n, {self.output_dim})"

        # Now, split params as specified by each parameter.dim 
        # and use their corresponding forward method.
        params_t = tuple()
        i = 0
        for p in self.params:
            j = i + p.dim
            p = p(params[:, i:j]) # pass activation function
            params_t += (p,)
            i = j

        return params_t
    
    def warm_start(self, x, *parents, theta=None):
        if hasattr(self, 'net') and hasattr(self.net, 'warm_start'):
            assert parents
            
            self.net.warm_start(torch.cat(parents, 1))
            
        return super().warm_start(x)


class Parameter(Module):
    """Class used to represent a DCN distribution's Parameter.

    Defines the name of the parameter, its dimensionality,
    an optional activation function to condition its domain
    (i.e., strictly positive parameters with a softplus + eps activation)
    and an optional initialization function (pre-activation).

    The activation function is defined in the forward method.
    """

    def __init__(self, name, dim=1):
        """
        Args:
            name (str): name of the parameter.
            dim (int): dimension of the parameter.
            init (function): initialization function(*shape) for the parameter.
                This is only used if Parameter is computed
                with a learnable, constant tensor (not conditioned by an input).
                If None, defaults to torch.randn.
        """

        super().__init__()

        self.name = name
        self.dim = dim

    def forward(self, x):
        return x # default
    
    def init(self):
        return None # by default, no initialization


# Parameter implementations:

class ParameterLoc(Parameter):
    r"""Location parameter in domain (a, b) (including infinity).

    Note that in all cases the interval is open, due to the effect
    of a small constant eps specified in the constructor.

    Every domain specifies a different activation function.
    """

    def __init__(self, name='loc', a=-np.inf, b=np.inf, eps=1e-3, **kwargs):
        super().__init__(name, **kwargs)

        self.a = a
        self.b = b
        assert 0 < eps and eps < .5
        self.eps = eps

        self.inf_a = np.isinf(a)
        self.inf_b = np.isinf(b)

        self.truncated = not (self.inf_a and self.inf_b)

    def forward(self, x):
        a, b = self.a, self.b
        inf_a, inf_b = self.inf_a, self.inf_b

        if self.truncated:
            if not inf_a and not inf_b:
                s = torch.sigmoid(x) * (1 - self.eps) + self.eps / 2.
                return s * (b - a) + a
            else:
                x = F.softplus(x) + self.eps

                if inf_a:
                    return b - x
                else: # inf_b
                    return a + x
        else:
            return x
        
    def init(self):
        a, b = self.a, self.b
        inf_a, inf_b = self.inf_a, self.inf_b
        
        if self.truncated:
            if not inf_a and not inf_b:
                return torch.ones(self.dim) * (b - a)
            else:
                return None
        else:
            return torch.zeros(self.dim)


class ParameterScale(ParameterLoc):

    def __init__(self, name='scale', eps=1e-3, **kwargs):
        super().__init__(name, a=0., eps=eps, **kwargs)
        
    def init(self):
        return softplus_inv(torch.ones(self.dim) - self.eps)


class ParameterLogScale(Parameter):
    r"""Log-scale parameter, with all reals as domain."""

    def __init__(self, name='logscale', **kwargs):
        super().__init__(name, **kwargs)

    def forward(self, x):
        return x
    
    def init(self):
        return torch.zeros(self.dim)
    


class ParameterShape(ParameterScale):

    def __init__(self, name='shape', eps=1e-3, **kwargs):
        super().__init__(name, eps=eps, **kwargs)


class Parameter01(ParameterLoc):

    def __init__(self, name, eps=1e-3, **kwargs):
        super().__init__(name, a=0., b=1., eps=eps, **kwargs)


class ParameterCategorical(Parameter):
    """Parameter for a Categorical distribution, log p."""

    def __init__(self, name='log_p', dim=2, **kwargs):
        assert dim >= 2

        super().__init__(name, dim=dim, **kwargs)

    def forward(self, x):
        return torch.log_softmax(x, dim=1)


class ParameterBernoulli(Parameter):
    """Parameter for a Bernoulli distribution, log p pre-activation.

    Returns the parameter tensor pre-activation. This is needed,
    since the input dimension will be 1 but the output will be 2.

    When using this parameter, call parameter.activation(x).
    This returns a 2-dim tensor, containing log((1 - p, p)).
    By returning two dimensions, we can use the Categorical functions. 
    """

    def __init__(self, name='_log_p', **kwargs):
        assert kwargs.get('dim', 1) == 1
        super().__init__(name, **kwargs)

    # Note that forward won't do anything

    def activation(self, x):
        assert x.size(1) == 1 # we'll transform it to 2
        
        return torch.cat([
            F.logsigmoid(-x), # log(1 - sigmoid(x)) = log sigmoid(-x)
            F.logsigmoid(x)
        ], 1)