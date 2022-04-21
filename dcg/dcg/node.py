"""Includes an abstract class for `CausalNode`.
"""

import torch
from torch import nn

from .utils import Module, requires_init


class CausalNode(Module):
    """Base class for a CausalNode.

    Implements basic functionality for any descendant class
    and defines the required attributes and methods to implement.

    Any inheriting class must implement the following methods:
    ```python
        def _warm_start(self, x, *_parents, theta=None):
            # Warm starts the node with some values x, given parents. 
            # Note that _parents includes ex_noise.
            # Optional, default doesn't do anything.
            ...
        
        def _sample(self, n, *_parents, theta=None):
            # Create n samples of this node's distribution, given parent values.
            # Note that _parents includes ex_noise.
            ...

        def _loglk(self, x, *_parents, theta=None):
            # Compute the Log-Likelihood of samples x, given parents.
            # Note that _parents includes ex_noise (which can be ignored here).
            ...

        def _abduct(self, x, *_parents, theta=None):
            # Abduct ex_noise from x and parents.
            # Note that _parents includes ex_noise (which can be ignored here).
            ...
    ```

    define class attributes:
        - latent (bool): whether the node is latent.
        - discrete (bool): whether the corresponding r.v. is discrete.
        
    and define instance attributes:
        - ex_noise (tuple): tuple of exogenous noise nodes.
        - ex_invertible (bool): whether the exogenous noise signals 
            can be inverted exactly (True) or just sampled (False).
            False by default.
        - theta_dim (int): number of external parameters 
            (total dimensionality of the theta parameter).
        - theta_init (Tensor): tensor of shape (theta_dim,) 
            with the initialization value for external theta_attributes.
            None if theta_dim is 0 or if no initialization is to be provided.
        
    Note that _sample, _loglk and _abduct may receive a theta attribute.
    If the node has its own network, this parameter should be ignored,
    but if it doesn't, theta should contain the required parameters for the node.
        
    Additionally, assign the attribute `theta_dim` with 0 if 
    the node can learn its own theta parameters (i.e., net_f is not None);
    otherwise, the number of parameters required (the total dimensionality of theta).
    """

    latent = None
    """Whether this node class is latent."""
    discrete = None
    """Whether this node class represents a discrete random variable."""


    def __init__(self, name, *parents, dim=1, **kwargs):
        """
        Args:
            name (str): name of the node.
            *parents (CausalNode): nodes that act as parents for this node.
            dim (int): dimensionality of the r.v. described by this node.

        Note that any remaining **kwargs are ignored,
        so that one can pass the same kwargs to all nodes,
        even those that don't require some of them."""

        super().__init__() # first call Module init
        
        assert isinstance(name, str) and name, self
        self.name = name

        # Set basic attributes and asserts        
        assert isinstance(dim, int) and dim >= 1, (self, dim)
        self.dim = dim

        # Note that parents is a tuple, so that they won't be saved as submodules.
        # This is important to avoid calling .to(device) repeatedly.
        # The graph will store all the visible nodes, 
        # no need for the nodes to do so.
        self.parents = parents
        
        self._ex_noise = None
        self._ex_invertible = False # False by default
        
        assert isinstance(self.discrete, bool), \
            f'Subclasses of CausalNode must define attribute discrete: {self}'
        assert isinstance(self.latent, bool), \
            f'Subclasses of CausalNode must define attribute latent: {self}'
        
        self.theta_dim = 0 # by default, unless overwritten
        self.theta_init = None # by default, unless overwritten

    @property
    def _parents(self):
        return tuple(self.parents) + tuple(self.ex_noise)
    
    @property
    def ex_noise(self):
        assert self._ex_noise is not None, \
            f'Must set ex_noise when instantiating node {self}.'
        
        return self._ex_noise
    
    @ex_noise.setter
    def ex_noise(self, ex):
        from collections.abc import Iterable
        if ex is None:
            ex = tuple()
        elif isinstance(ex, Iterable):
            ex = tuple(ex)
        else:
            ex = (ex,)
            
        assert all(not e.parents for e in ex), \
            f'All ex nodes must be parentless: {name}'
        assert len(set(e.name for e in ex)) == len(ex), \
            f'ex_noise had not unique names: {name}'

        self._ex_noise = nn.ModuleList(ex)

    @property
    def theta_init(self):
        return self._theta_init
    
    @theta_init.setter
    def theta_init(self, value):
        assert not self.theta_dim or value is None or value.shape == (self.theta_dim,)
        self._theta_init = value
    
    @property
    def trainable(self):
        """Whether this node class is trainable."""

        # If a child class has set its trainable attribute, we take that value.
        # Otherwise, being trainable means having any trainable parameter.
        if not hasattr(self, '_trainable'):
            self._trainable = None

        if self._trainable is None:
            return any(p.numel() for p in self.parameters())
        else:
            return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        
        
    def _process_parents(self, n, parents):
        """Process all incoming parents to fulfill the requirements.

        Checks that all passed parents have appropriate shape 
        and broadcasts samples if required + (n > p.size(0)).

        Also samples from self.ex_noise if not present in *parents.
        
        Returns two tuples, one with all parents, one with all ex_noises.
        Can be overriden to return any number of objects, 
        but make sure that the result is iterable 
        so as to pass it to the rest of the methods like *result.
        (Example, ```return tensor,```, which returns a tuple (tensor,)).
        """

        # Check that all parents are passed
        assert len(parents) >= len(self.parents) and \
            all(p is not None for p in parents[:len(self.parents)]), \
            f'Not all parents specified: {self}'

        # Check that they are tensors with the appropriate shape
        assert all(
            p is None or (
                isinstance(p, torch.Tensor) and 
                len(p.shape) == 2 and 
                not n % p.size(0) and 
                p.size(1) == p_node.dim
            )

            for p, p_node in zip(parents, self._parents)
        ),  f'Incorrect parents shape: {self} ' + str({ 
                p.name: tuple(t.shape) 
                for p, t in zip(self._parents, parents) 
            })

        # Broadcast all tensors
        parents = list(
            p.repeat(n // p.size(0), 1) if p is not None else p 
            for p in parents
        )

        # Sample from ex_noise if required
        for i, ex in enumerate(self.ex_noise, len(self.parents)):
            if len(parents) <= i or parents[i] is None:
                sample = ex.sample(n)

                if i < len(parents):
                    parents[i] = sample
                else:
                    parents.append(sample)
                    
        return tuple(parents)

        
    # To override:
    def warm_start(self, x, *parents, theta=None):
        """Warm start the node with some values x given parents."""
        _parents = self._process_parents(x.size(0), parents)
        self._warm_start(x, *_parents, theta=theta)
        
        return super().warm_start(x)
    
    def _warm_start(self, x, *_parents, theta=None):
        return # default behaviour

        
    @requires_init    
    def sample(self, n, *parents, theta=None):
        """Sample from this node's r.v., conditioned on its parents.

        Args:
            n (int): number of instances to sample.
            *parents (torch.Tensor): parents values.
            theta (torch.Tensor): Tensor with external parameters.
                None if there's not any.
        """
        _parents = self._process_parents(n, parents)
        
        return self._sample(n, *_parents, theta=theta)
        
    def _sample(self, n, *_parents, theta=None):
        raise NotImplementedError()

        
    @requires_init
    def loglk(self, x, *parents, theta=None):
        """Negative Log-Likelihood of tensor x conditioned on its parents.

        Args:
            x (torch.Tensor): observable values for the node.
            *parents (torch.Tensor): parent values.
            theta (torch.Tensor): Tensor with external parameters.
                None if there's not any.
        """
        _parents = self._process_parents(x.size(0), parents)
        
        return self._loglk(x, *_parents, theta=theta)
        
    def _loglk(self, x, *_parents, theta=None):
        raise NotImplementedError()
        
        
    def nll(self, *args, **kwargs):
        return -self.loglk(*args, **kwargs)

    
    @requires_init
    def abduct(self, x, *parents, theta=None):
        """Sample ex_noise conditioned on x and its parents.

        Args:
            x (torch.Tensor): observable values for the node.
            *parents (torch.Tensor): parent values.
            theta (torch.Tensor): Tensor with external parameters.
                None if there's not any.
        """
        _parents = self._process_parents(x.size(0), parents)
        
        return self._abduct(x, *_parents, theta=theta)
        
    def _abduct(self, x, *_parents, theta=None):
        raise NotImplementedError()


    # Utils
    @property
    def depth(self):
        if not hasattr(self, '_depth'):
            # Returns topological order depth. 
            # Note that this requires the graph to be acyclic.
            if not self.parents:
                self._depth = 0
            else:
                self._depth = max(parent.depth for parent in self.parents) + 1
            
        return self._depth


    # Magic methods
    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        if hasattr(self, 'name'):
            return f'<{self.__class__.__name__}: {self.name}>'
        else:
            # Non-initialized node (yet)
            return '<{self.__class__.__name__}>'

    def __lt__(self, other):
        return self.name < other.name # alphabetical order

    def __iter__(self):
        """Yield all its ex_noises and self. 
        
        Used to get the topological ordering of ALL nodes in a graph,
        if each node's __iter__ is called in topological ordering.

        Example:
        for node in topological_ordering(graph.nodes):
            for subnode in node:
                yield subnode
        """

        for ex in self.ex_noise:
            yield ex

        yield self


    # Device methods
    # We override them to save the corresponding device 
    def _update_device(self, device):
        super()._update_device(device)

        # Include all subnodes (since they are stored in a nn.ModuleList)
        for node in self.ex_noise:
            node.device = device