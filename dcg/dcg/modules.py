"""
This module provides miscellaneous nn.Modules and CausalNodes.

* `InputNode`
"""

import numpy as np
import torch
from torch import nn

from .node import CausalNode
from .distributional.discrete import Bernoulli
from .latents import Gumbel
from .sampling import categorical
from .utils import log_sum_exp_trick, requires_init


class InputNode(CausalNode):
    """Direct input Causal Node.

    Useful to include input tensors 
    directly to the causal graph as interventions.

    Its loglk will always be 0 to ignore its effects (just like interventions),
    and doesn't have implementations for sample and abduct. 
    """

    latent = False
    # discrete is defined as a constructor parameter

    def __init__(self, name, *parents, discrete=False, **kwargs):
        """
        Args:
            name (str): name of the node.
            *parents (CausalNode): parents of the node.
            discrete (bool): whether the node is discrete or continuous.

        Note that even if this node may have parents, 
        it should always come as an intervention to the graph.
        """

        object.__setattr__(self, 'discrete', discrete) # set before init

        super().__init__(name, *parents, **kwargs)
        
        self.ex_noise = []
        self.ex_invertible = True

    # Overrides
    def _sample(self, n, *_parents, theta=None):
        # Should not be called, since its values always come from intervention.
        raise NotImplementedError()

    def _loglk(self, x, *_parents, theta=None):
        return torch.zeros_like(x[:, 0])

    def _abduct(self, x, *_parents, theta=None):
        # Should not be called, since its values always come from intervention.
        raise NotImplementedError()


class MixtureNode(CausalNode):
    
    latent = False
    
    @property
    def requires_initialization(self):
        if not hasattr(self, '_requires_initialization'):
            self._requires_initialization = any(
                node.requires_initialization 
                for node in self.mixture_nodes
            )
            
        return self._requires_initialization

    @staticmethod
    def _check_mixture_nodes(mixture_nodes, parents=None):
        assert len(mixture_nodes)
        unique = lambda f, it: len(set(f(x) for x in it)) == 1

        if parents is None:
            parents = mixture_nodes[0].parents
        
        assert (
            isinstance(mixture_nodes, (tuple, list, set)) and
            mixture_nodes and
            unique(lambda x: x.name, mixture_nodes) and
            all(
                len(parents) == len(node.parents) and all(
                    p1.name == p2.name
                    for p1, p2 in zip(parents, node.parents)
                )
                for node in mixture_nodes
            ) and 
            unique(lambda x: x.discrete, mixture_nodes) and
            unique(lambda x: len(x.ex_noise), mixture_nodes) and
            all(
                unique(lambda x: (x.name, x.dim, x.discrete), ex)
                for nodes in zip(*mixture_nodes)
                for ex in zip(*map(lambda node: node.ex_noise, nodes))
            )
        )

    @classmethod
    def from_nodes(cls, mixture_nodes, parents, w=None, **kwargs):
        cls._check_mixture_nodes(mixture_nodes, parents)

        n = mixture_nodes[0]
        return cls(
            n.name, *parents, 
            dim=n.dim, mixture_nodes=mixture_nodes, w=w, **kwargs
        )
    
    def __init__(self, name, *parents, dim=1, n_components=10, component=None, mixture_nodes=None, w=None, **kwargs):
        assert mixture_nodes is not None or (component is not None and n_components > 0)
        if mixture_nodes is None:
            if not isinstance(component, (list, tuple)):
                component = [component] * n_components
            else:
                assert len(component) == n_components
                
            mixture_nodes = [
                c(name, *parents, dim=dim, **kwargs)
                for c in component
            ]
        else:
            n_components = len(mixture_nodes)

        self._check_mixture_nodes(mixture_nodes, parents)
        
        object.__setattr__(self, 'discrete', mixture_nodes[0].discrete)
        
        super().__init__(name, *parents, dim=dim, **kwargs)
        
        self.mixture_nodes = nn.ModuleList(mixture_nodes)
        self.K = len(mixture_nodes)
        
        self.ex_noise = (
            (Gumbel(self.name + '.gumbel', dim=self.K),) +
            tuple(self.mixture_nodes[0].ex_noise)
        )

        self.ex_invertible = False
        
        # Replace mixture_nodes elements
        for node in mixture_nodes:
            node.ex_noise = self.ex_noise[1:]
            node.parents = parents
        
        if w is None:
            w = torch.randn(self.K)
        else:
            assert (
                isinstance(w, torch.Tensor) and 
                w.flatten().shape == (self.K,)
            )
            
        self._w = nn.Parameter(w.view(1, self.K))
        
        self.theta_dim = sum(node.theta_dim for node in self.mixture_nodes)
        theta_init = [ 
            node.theta_init 
            if node.theta_init is not None
            else torch.randn(node.theta_dim) 
            
            for node in self.mixture_nodes 
            if node.theta_dim 
        ]
        self.theta_init = torch.cat(theta_init) if theta_init else None
        
    @property
    def log_w(self):
        return torch.nn.functional.log_softmax(self._w, 1)
        
    # To override:
    def _update_device(self, device):
        super()._update_device(device)
        
        for node in self.mixture_nodes:
            node.device = device
    
    def _process_parents(self, n, *parents):
        _parents = super()._process_parents(n, *parents)
        
        parents, ex = _parents[:len(self.parents)], _parents[len(self.parents):]
        gumbel, ex = ex[0], ex[1:]
        
        return parents, gumbel, ex
    
    def _process_theta(self, theta):
        if theta is None:
            return (None,) * self.K
        else:
            res = []
            i = 0
            for node in self.mixture_nodes:
                j = i + node.theta_dim
                if j - i:
                    res.append(theta[:, i:j])
                else:
                    res.append(None)
                i = j
            
            return tuple(res)

    def _warm_start(self, x, parents, gumbel, ex, theta=None):
        """Warm start the node with some values x."""
        theta = self._process_theta(theta)
        
        for node, theta_i in zip(self.mixture_nodes, theta):
            node.warm_start(x, *parents, *ex, theta=theta_i)

    def _sample(self, n, parents, gumbel, ex, theta=None):
        """Sample from this node's r.v., conditioned on its parents.

        Args:
            n (int): number of instances to sample.
            *parents (torch.Tensor): parents values.
        """
        theta = self._process_theta(theta)
        components = categorical(n, self.log_w, gumbel)
        
        return (
            components.unsqueeze(2) * 
            torch.stack([
                node.sample(n, *parents, *ex, theta=theta_i)
                for node, theta_i in zip(self.mixture_nodes, theta)
            ], 1)
        ).sum(1)

    def _loglk(self, x, parents, gumbel, ex, theta=None):
        """Negative Log-Likelihood of tensor x conditioned on its parents.

        Args:
            x (torch.Tensor): observable values for the node.
            *parents (torch.Tensor): parent values.
        """
        theta = self._process_theta(theta)
        
        return log_sum_exp_trick(
            torch.stack([
                node.loglk(x, *parents, theta=theta_i)
                for node, theta_i in zip(self.mixture_nodes, theta)
            ], 1),
            log_w=self.log_w
        )

    def _abduct(self, x, parents, gumbel, ex, theta=None):
        """Sample ex_noise conditioned on x and its parents.

        Args:
            x (torch.Tensor): observable values for the node.
            *parents (torch.Tensor): parent values.
        """
        theta = self._process_theta(theta)
        components = categorical(x.size(0), self.log_w, gumbel)
        
        return (gumbel,) + tuple(
            (
                components.unsqueeze(2) * 
                torch.stack([ ex for ex in ex_t ], 1)
            ).sum(1)
            for ex_t in zip(*(
                node.abduct(x, *parents, *ex, theta=theta_i)
                for node, theta_i in zip(self.mixture_nodes, theta)
            ))
        )