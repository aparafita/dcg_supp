"""
This module provides the FlowNode,
an implementation of a DCU using Normalizing Flows.
"""

import torch
from torch import nn

from .node import CausalNode
from .latents import LatentNode


class LatentFlowPrior(LatentNode):
    """Latent causal node from a flow prior."""
    
    discrete = False # by default, unless re-specified in __init__

    def __init__(self, *args, prior=None, **kwargs):
        assert prior is not None
        dim = kwargs.get('dim', prior.dim)
        assert dim == prior.dim
        
        super().__init__(*args, dim=dim, **kwargs)
        
        self.prior = prior
        self.discrete = prior.discrete

    # To override:
    def _sample(self, n, theta=None):
        """Sample n rows from the node's distribution and send to device."""
        return self.prior.sample(n)

    def _loglk(self, x, theta=None):
        """Compute the nll of x."""
        return self.prior.loglk(x)
    
    
class NCF(CausalNode):
    """Normalizing Causal Flow.

    https://arxiv.org/abs/2006.08380

    Uses Normalizing Flows as the Causal Mechanism for a SCM.
    """
    
    latent = False
    discrete = False
    
    @property
    def requires_initialization(self):
        return self.flow.requires_initialization

    @property
    def undiscretize(self):
        return self._undiscretize.item()
    
    def __init__(
        self, 
        name, *parents, dim=1, flow_f=None, head_parents=None,
        undiscretize=False, undiscretize_sigma=.5 / 1.96,
        **kwargs
    ):
        """
        Args:
            name (str): name of the node.
            *parents (CausalNode): nodes that act as parents for this node.
            dim (int): dimensionality of the r.v. described by this node.
            flow_f (function): function(dim=1, cond_dim=0) that returns 
                a conditional Flow with these dimensions. 
                If head_parents is not None, 
                flow is a function(dim=1, cond_dim=0, head_slices=[])
                that receives the list of slices needed for using the MultiHeads.
            undiscretize (bool): whether to undiscretize discrete signals
                (by adding Normal noise) so as to learn ordinal variables
                as continuous variables.
            undiscretize_sigma (float): scale of the Normal noise
                added when undiscretizing.

        It is recommended that any flow used as a CausalNode 
        has BatchNorm as the initial step 
        so as to normalize the node's distribution.
        Also, this flow should be conditional (if theta_dim == 0)
        so as to take into account the node's parent values.
        """
        
        assert flow_f is not None, \
            'Remember to define the flow template for every NCF node'
        
        cond_dim = sum(p.dim for p in parents)
        
        if head_parents:
            head_slices = []
            head_dims = 0
            
            start, end = 0, 0
            for p in parents:
                end += p.dim
                
                if p.name in head_parents:
                    head_slices.append(slice(start, end))
                    head_dims += p.dim
                    
                start = end
            
            flow = flow_f(dim, cond_dim, head_slices=head_slices)
        else:
            flow = flow_f(dim, cond_dim)
            head_slices = None
        
        super().__init__(name, *parents, dim=dim)
        
        # Assign the two terms we computed before
        self.cond_dim = cond_dim
        self.head_slices = head_slices
        self.flow = flow
        
        self.theta_dim = flow.theta_dim
        self.theta_init = flow.theta_init()
        
        self.ex_noise = LatentFlowPrior(self.name + '.ex', prior=self.flow.prior)
        self.ex_invertible = True

        self.register_buffer('_undiscretize', torch.tensor(undiscretize))
        self.register_buffer('undiscretize_sigma', torch.tensor(undiscretize_sigma))


    # Overrides for CausalNode:
    def _process_parents(self, n, parents):
        # Override _process_parents to split parents and ex 
        # and stack each group in single tensors.
        _parents = super()._process_parents(n, parents)
        parents, ex = _parents[:-len(self.ex_noise)], _parents[-len(self.ex_noise):]

        parents = torch.cat(parents, 1) if parents else None
        ex = torch.cat(ex, 1) if ex else None

        return parents, ex
    
    def _sample(self, n, parents, ex, theta=None):
        # kwargs are passed to flow.invert
        sample = self.flow(ex, theta=theta, invert=True, cond=parents)

        return sample

    def _loglk(self, x, parents, ex, theta=None):
        if self.training and self.undiscretize:
            x = x + torch.randn_like(x) * self.undiscretize_sigma
        
        loglk = self.flow.loglk(x, theta=theta, cond=parents)

        return loglk
    
    def _abduct(self, x, parents, ex, theta=None):
        ex_x = self.flow(x, theta=theta, cond=parents)

        return (ex_x,)
    
    def _warm_start(self, x, parents, ex, theta=None):
        self.flow.warm_start(x, theta=theta, cond=parents)
        
        super()._warm_start(x, parents, ex, theta=theta)