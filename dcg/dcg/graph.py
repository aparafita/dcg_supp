"""
Contains `CausalGraph`, with all utilities for graph-related operations. 
"""

from copy import copy, deepcopy
from functools import partial

import warnings

import torch
from torch import nn

from .node import CausalNode
from .utils import topological_order, log_mean_exp_trick, AdjacencyMaskedNet, Module, requires_init


class CausalGraph(Module):
    """
    Provides utilities for easy creation of graphs and their nodes.
    Provides utilities for sampling, nll computation and counterfactual queries.
    """
    
    @property
    def requires_initialization(self):
        if not hasattr(self, '_requires_initialization'):
            self._requires_initialization = any(
                node.requires_initialization 
                for node in self._all_nodes
            )
            
        return self._requires_initialization
    
    # Construction methods:

    def __init__(self, nodes, net_f=None, head_nodes=None):
        """
        Args:
            nodes (list): list of nodes defining the graph.
            net_f (function): function(input_dim, output_dim, init=None) 
                that returns the network that computes external parameters theta.

        Note that CausalGraph only works for DAGs. 
        Additionally, all nodes must have unique names.
        """

        super().__init__()

        self.nodes = nn.ModuleList(nodes)

        parents = { node: node.parents for node in nodes }
        # self._nodes = topological_order(self.nodes, parents) # TODO: remove this
        try:
            self._nodes = sorted(self.nodes, key=lambda node: node.depth)
        except RecursionError:
            raise Exception('Cyclic graphs not allowed in CausalGraph')

        self._all_nodes = [
            subnode 
            for node in self._nodes 
            for subnode in node 
        ]

        assert len({node.name for node in self._all_nodes}) == \
            len(self._all_nodes), 'Node names must be unique'
        
        self.theta_dim = sum(node.theta_dim for node in self.nodes)
        theta_init = [ 
            node.theta_init 
            if node.theta_init is not None
            else torch.randn(node.theta_dim) 
            
            for node in self._nodes 
            if node.theta_dim
        ]
        self.theta_init = torch.cat(theta_init) if theta_init else None
        assert self.theta_init is None or self.theta_init.shape == (self.theta_dim,)
        
        head_nodes = list(set(head_nodes or set()))
        assert all(
            any(name == node.name for node in self.nodes)
            for name in head_nodes
        )
                      
        self.theta_slices = {}
        if self.theta_dim:
            assert net_f is not None
            
            # Let's define the adjacency matrix A
            A = torch.zeros(self.latent_dim + self.dim, self.theta_dim, dtype=bool)
            start = 0
            for node in self._nodes:
                end = start + node.theta_dim
                if end - start:
                    self.theta_slices[node] = slice(start, end)
                
                for parent in node.parents:
                    A[self.index(parent, latents=True), start:end] = True
                
                start = end
                
            head_slices = []
            for name in head_nodes:
                head_slices.append(self.index(self[name], latents=True))
                                
            if head_slices:
                net_f = partial(net_f, head_slices=head_slices)
            
            self.net = AdjacencyMaskedNet(A, net_f=net_f, init=self.theta_init)
        else:
            self.net = None

    @staticmethod
    def parse_definition(definition, **cls_dict):
        """Parse a graph definition for use in from_definition.

        A graph definition is a list of 4-tuples:
            `(node_name, node_class, node_dim, node_parents)`
        where:

        * node_name (str): name of the node. Must not contain '.'
            and must be unique across the whole definition.
        * node_class (str or CausalNode class): which CausalNode class
            to assign to this node. If str, the method looks in **cls_dict
            for the corresponding class.
        * node_dim (int): dimension of the node.
        * node_parents (tuple of str): which nodes (indexed by their name)
            are this node's parents.

        A graph definition can also be stated as a multi-line str 
        in the following format:
        ```
            node_name1 node_class node_dim parent1 parent2
            node_name2 ...
        ```

        Example:
        ```
            u latent_normal 1
            x normal 1 u
            y categorical 2 u
        ```

        Args:
            definition: str or list of 4-tuples with the graph definition.
            **cls_dict (class): 
                use keyword arguments directly to relate
                class aliases to their CausalNode classes.
                These will be packed in a dict inside the method.

        The result of this function can be passed to 
        `CausalGraph.from_definition` directly to create the graph.

        Call example:
        ```python
        from dcg import latents, distributional, flow

        definition = CausalGraph.from_definition(
            '''
            u latent 1
            x normal 1 u
            y categorical 2 u
            z flow 1 x y
            '''.
            latent=latents.Normal,
            normal=distributional.continuous.Normal,
            categorical=distributional.discrete.Categorical,
            flow=flow.NCF
        )
        ```
        """

        if isinstance(definition, str):
            definition = [
                [t[0], t[1], t[2], t[3:]]

                for t in (
                    [ term.strip() for term in line.strip().split(' ') ]
                    for line in definition.split('\n')
                    if line.strip()
                )
            ]

        for line in definition:
            _, cls, dim, _ = line
            
            if isinstance(cls, str):
                line[1] = cls_dict[cls]

            if isinstance(dim, str):
                line[2] = int(dim)

        return definition

    @classmethod
    def from_definition(
        cls, definition, node_kwargs=None, global_kwargs=None, net_f=None, head_nodes=None, **kwargs
    ):
        """Create a CausalGraph from a list of tuples (name, cls, dim, parents).

        You can use `CausalGraph.parse_definition` to create a graph definition.

        Args:
            node_kwargs (dict or None): dictionary { node_name: kwargs }
                used to pass keyword arguments as **kwargs 
                to the node with the corresponding node_name. 
            global_kwargs (dict or None): dictionary used to pass keyword arguments as **global_kwargs
                to all nodes. node_kwargs can override any kwarg in global_kwargs.
            net_f (function or None): function (input_dim, output_dim, init=None) 
                that creates the network that will compute the external parameters theta.
            head_nodes (list or None): if using multiheaded nodes, pass which nodes act as splitters.
            **kwargs: kwargs passed as **kwargs to the graph's constructor.
                Note that you can pass these kwargs directly to the method,
                in contrast with node_kwargs and global_kwargs.
        """
        
        # Assert no repetitions in naming
        node_names = list(map(lambda t: t[0], definition))
        assert len(set(node_names)) == len(node_names), \
            'Node names must be unique'

        # Order definition by topological order (we need it for Node creation)
        original_order = [ line[0] for line in definition ]
        definition_d = { line[0]: line for line in definition }

        parents = { line[0]: set(line[3]) for line in definition_d.values() }
        new_order = topological_order(original_order, parents)

        # Reorder definition by topological ordering defined by new_order
        definition = [ definition_d[name] for name in new_order ]

        # Create final node_kwargs dicts
        node_kwargs = deepcopy(node_kwargs or {})
        global_kwargs = global_kwargs or {}
        for name, node_cls, dim, parents in definition:
            if name in node_kwargs:
                assert isinstance(node_kwargs[name], dict), name
            else:
                node_kwargs[name] = {}

            node_kwargs[name].update({ 
                k: v
                for k, v in global_kwargs.items()
                if k not in node_kwargs[name]
            })
            
        # Check if we need to pass head_nodes to any node
        head_nodes = list(set(head_nodes or set()))
        assert all(
            any(name == node_name for node_name, _, _, _ in definition)
            for name in head_nodes
        )

        # Now, create nodes in topological order
        d = {}

        for name, node_cls, dim, parents in definition:
            parents = tuple(d[parent] for parent in parents)
            d[name] = node_cls(
                name, *parents, dim=dim, **node_kwargs[name], 
                head_parents=[
                    name
                    for name in head_nodes
                    if any(name == p.name for p in parents)
                ]
            )

        # Finally, reorder nodes by the original ordering
        nodes = [ d[name] for name in original_order ]

        return cls(nodes, **kwargs, net_f=net_f, head_nodes=head_nodes)
    

    def replace(self, original, new_node):
        """Replace original node by new_node.
        
        Adjusts parents and every relevant attribute in the node and in the graph.
        """
        original = self[original] # in case it's an indexer

        def replace(l):
            if isinstance(l, tuple):
                i = l.index(original)
                l = l[:i] + (new_node,) + l[i+1:]
            else:
                # if it's modulelist, we need it as tuple for .index
                i = tuple(l).index(original) 
                l[i] = new_node

            return l

        for node in self._all_nodes:
            if original in node.parents:
                node.parents = replace(node.parents)

        replace(self.nodes)
        replace(self._nodes)

        self._all_nodes = [
            subnode 
            for node in self._nodes 
            for subnode in node 
        ]

        new_node.to(self.device)
        
        if original in self.theta_slices:
            self.theta_slices[new_node] = self.theta_slices[original]
            del self.theta_slices[original]
        
        return self
    

    # Convenience attributes and methods
    def ancestors(self, node, all=False, visited=None):
        """Iterator that yields node and all its ancestors.

        Args:
            node (CausalNode): node to get its ancestors from.
            all (bool): whether to also yield latent private nodes or not.
        """
        root = visited is None
        if root:
            visited = set()
            
        visited.add(node)
            
        for parent in (node.parents if not all else node._parents):
            if parent not in visited:
                for _ in self.ancestors(parent, all=all, visited=visited):
                    pass
               
        if root:
            for node in visited:
                yield node

    def descendants(self, node, all=False):
        """Iterator that yields node and all its descendants.

        Args:
            node (CausalNode): node to get its descendants from.
            all (bool): whether to also yield latent private nodes or not.
        """

        assert node in self._all_nodes, node

        for descendant in (self.nodes if not all else self._all_nodes):
            if node in self.ancestors(descendant, all=all):
                yield descendant
                
    def depth_levels(self, all=False):
        nodes = self._nodes if not all else self._all_nodes
        
        levels = []
        for node in nodes:
            depth = node.depth
            while len(levels) <= depth:
                levels.append([])
                
            levels[depth].append(node)
        
        return levels

    @property
    def node_names(self):
        """Return the list of node names."""
        return [ node.name for node in self.nodes ]

    @property
    def observable_nodes(self):
        """Return the list of observable nodes."""
        return [ node for node in self.nodes if not node.latent ]

    @property
    def latent_nodes(self):
        """Return the list of latent nodes."""
        return [ node for node in self.nodes if node.latent ]

    @property
    def trainable_nodes(self):
        """Return the list of (all) trainable nodes."""
        return [ 
            node 

            for node in self._all_nodes 
            if node.trainable
        ]

    @property
    def dim(self):
        """Reutrn the total observable dimension of the graph."""
        return sum(node.dim for node in self.observable_nodes)
    
    @property
    def latent_dim(self):
        return sum(node.dim for node in self.latent_nodes)
    
    def index(self, key, latents=False):
        """Return a `slice` with the tensor columns related with this node."""
        target_node = self[key] # if node is str, replaces by node
        assert target_node in self.nodes and (latents or not target_node.latent), target_node
        # note that latents do not have an index

        i = 0
        for node in (self.nodes if latents else self.observable_nodes):
            j = i + node.dim
            if target_node == node:
                return slice(i, j)
            else:
                i = j

        raise KeyError(key)

    def tensor_to_dict(self, t, latents=False):
        """Transform a tensor with values for all observable nodes in the graph
        to a dict { node: value }.
        """

        dim = self.dim if not latents else (self.latent_dim + self.dim)
        assert len(t.shape) == 2 and t.size(1) == dim, (t.shape, dim)

        d = { 
            node: t[:, self.index(node, latents=latents)]
            for node in (self.observable_nodes if not latents else self.nodes)
        }

        # Change all dtypes to float and moves to device
        d = self._preprocess_dict(d, t.size(0))

        return d

    def dict_to_tensor(self, d, latents=False):
        """Transform a dict { node: value } into a tensor."""
        d = self._preprocess_dict(d)

        return torch.cat([
            d[node]

            for node in (self.nodes if latents else self.observable_nodes)
            # if node in d # should contain all observable nodes
        ], dim=1)

    def _preprocess_dict(self, d, n=None):
        """Preprocess dict { node: value } for use in internal functions.

        This function transforms any kind of data into floats 
        (so that you can pass even bools as node value's)
        and non-tensors into tensors,
        shaping them correctly and even broadcasting samples if necessary. 

        If n is given, broadcasts all size(0) to that n. 
        If None, n becomes the maximum size(0) in d.
        """

        def to_tensor(v, node):
            # Redimension values with only one dim so they have 2.
            if not isinstance(v, torch.Tensor):
                if isinstance(v, (bool, int, float)):
                    v = [v]
                    
                v = torch.Tensor(v)

            while len(v.shape) < 2:
                assert node.dim == 1 # only allowed for 1-dimensional nodes
                v = v.unsqueeze(1 if len(v.shape) > 0 else 0)

            assert len(v.shape) == 2

            # Change dtype and move to device
            v = v.float().to(self.device)

            return v

        # Note that this (shallow-)copies the dict, as intended
        d = { 
            self[node]: to_tensor(v, self[node]) # also node str -> Node
            for node, v in d.items() 
        }

        if n is None: # n = max(size(0))
            n = max((v.size(0) for v in d.values()), default=1)

        # Need to be sure that we can broadcast (size(0) divisor of n)
        assert all(not n % v.size(0) for v in d.values()), \
            f'Not all values in d match with size {n}'

        # Broadcast
        d = { node: v.repeat(n // v.size(0), 1) for node, v in d.items() }

        return d

    def map(self, f, filter=None, all=False):
        """Call f(node) for all nodes in graph.

        Args:
            f (function): function(node) to call for each node.
            filter (function): function(node) that returns 
                whether node should be called. If None, no filter is applied.
            all (bool): whether to also call private subnodes.
        """

        for node in self.nodes if not all else self._all_nodes:
            if filter is None or filter(node):
                f(node)

    def warm_start(self, x):
        """Call warm_start to all nodes with their given value.

        Args:
            x: tensor or dict with all observable values,
                that will be passed to each corresponding node.
        """
        if isinstance(x, torch.Tensor):
            d = self.tensor_to_dict(x)
        else:
            assert isinstance(x, dict)
            d = self._preprocess_dict(x)
            
        # Sample latents if missing
        n = d[self.observable_nodes[0]].size(0)
        for node in self.latent_nodes:
            if node not in d:
                d[node] = node.sample(n)
            
        assert all(node in d for node in self.nodes)
            
        if self.net is not None:
            x_ = torch.cat([d[node] for node in self._nodes], 1)
            self.net.warm_start(x_)
            theta = self.net(x_)
        else:
            theta = None

        for node, x in d.items():
            theta_node = (
                theta[:, self.theta_slices[node]] 
                if theta is not None and node in self.theta_slices 
                else None
            )
            node.warm_start(x, *(d.get(p) for p in node.parents), theta=theta_node)

        return super().warm_start(x)
    
    # TODO: Remove this
#     def subgraph(self, nodes):
#         node_names = [ node.name for node in self._all_nodes ]
#         assert isinstance(nodes, (tuple, list, set)) and all(
#             isinstance(node, str) and node in node_names
#             for node in nodes
#         )

#         nodes = [ self[node] for node in nodes ] # respect order -> list
#         assert all(
#             ancestor in nodes
#             for node in nodes
#             for ancestor in self.ancestors(node)
#         )

#         return CausalGraph(nodes).to(self.device)


    # Magic methods (iterability and key-access)
    def __iter__(self):
        # Note that we iter through the original ordering of the nodes.
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes) # number of user-defined nodes (not private)

    def __getitem__(self, key):
        """Look for node in the graph, indexed by key, and return it.
        
        Depending on the type of key, the result differs:
            * `str`: node with that name.
            * `int`: node in that position in the original order.
            * `slice`: list of nodes that match the slice.
            * `CausalNode`: returns the same node, if it belongs to the graph.
                Otherwise, raises a KeyError. This is used to transform 
                a name str into a node, even when you don't know 
                if your key is a str or the node you want directly.
        """

        if isinstance(key, str): # index by name
            for node in self._all_nodes:
                if node.name == key:
                    return node
            else:
                raise KeyError(key)
        elif isinstance(key, (int, slice)): # index by position in original ordering
            return self.nodes[key]
        else:
            # it should be a node that's contained in the graph
            if key in self._all_nodes:
                return key
            else:
                raise KeyError(key)

    def _update_device(self, device):
        super()._update_device(device)
        
        # Add all nodes as they are in a nn.ModuleList
        for node in self.nodes:
            node.device = device

    # Basic operations
    @requires_init
    def sample(
        self, 
        n=1, target_node=None, interventions=None,
        return_all=False
    ):
        """Sample from graph.

        Args:
            n (int): how many samples to sample.
            target_node (str or CausalNode): 
                if not None, just sample ancestors of target_node 
                and return only its value.
            interventions (dict): what interventions to apply, if any.
                Dict { node: value } with the constant value to apply 
                to each of the n samples. You can pass python scalar values,
                even booleans, or single dim tensors,
                and they will be transformed and broadcasted to all n samples.
            return_all (bool): whether to return a dict with values 
                for every node, including latents.

        Returns:
            t: tensor (n, graph.dim) with samples for all observable nodes.
                This will be returned in both cases of `return_all`.
            d: dict { node: value} for all nodes, including latents,
                only if `return_all` is True. The result will then be `(t, d)`.
        """
        assert isinstance(n, int) and n >= 1

        if target_node is not None:
            target_node = self[target_node] # transform str to Node
            ancestors = list(self.ancestors(target_node, all=True))

        # Prepare interventions
        assert isinstance(interventions, dict) or interventions is None
        interventions = self._preprocess_dict(interventions or {}, n=n)

        x = {
            node: v
            for node, v in interventions.items()
        } # this will contain each node's samples
        
        levels = self.depth_levels(all=True) # in topological order, all nodes (including ex_noise)
        x_ = torch.zeros(n, self.latent_dim + self.dim, device=self.device)
        for level in levels:            
            if self.net is not None:
                # Compute parameters for this level                    
                subidx = torch.zeros(self.net.output_dim, dtype=bool, device=self.device)
                for node in level:
                    sl = self.theta_slices.get(node)
                    if sl is not None:
                        subidx[sl] = True
                
                theta = self.net(x_, subidx=subidx)
                theta_d = {}
                
                start = 0
                for node in self._nodes:
                    if node in level and node in self.theta_slices:
                        end = start + node.theta_dim
                        if end - start:
                            theta_d[node] = theta[:, start:end]
                            start = end
                            
                theta = theta_d
            else:
                theta = {}
                
            # Sample from all nodes in this level
            for node in level:
                if (target_node is not None and node not in ancestors):
                    continue

                xv = x.get(node)
                if xv is None:
                    mx = torch.ones((n,), dtype=bool, device=self.device)
                else:
                    mx = torch.isnan(xv).any(1)

                if mx.any().item():
                    sample = node.sample(
                        n, *(x[p] for p in node._parents), 
                        theta=theta.get(node)
                    )

                    if xv is None:
                        x[node] = sample
                    else:
                        xv[mx] = sample[mx]
                        
                if node in self.nodes:
                    x_[:, self.index(node, latents=True)] = x[node]

        # Transform x to tensor
        if target_node is None:
            t = self.dict_to_tensor(x) # transform the whole dict into a tensor
            # Note that this does only include non-latent nodes!
        else:
            t = x[target_node] # just get the tensor for the target_node

        if return_all:
            return t, x
        else:
            return t

    def _preprocess_x(self, x):
        """Tranform x into dict, ensures that sizes match and return (x, n).

        Used by loglk and counterfactual.
        """

        if isinstance(x, torch.Tensor):
            assert x.device.type == self.device.type
            n = x.size(0)
            x = self.tensor_to_dict(x)
        else:
            assert isinstance(x, dict)
            x = self._preprocess_dict(x)
            n = max((v.size(0) for v in x.values()), default=1)
            # Note that each node's tensor may have less than n samples.
            # This is no problem, since node._process_parents 
            # already does the broadcasting.

        x = copy(x) # make a copy of the dict

        return x, n
    
    def _preprocess_fX(self, f, X):
        if f is not None:
            if not isinstance(f, (list, tuple)):
                f = [f]

            l = []
            for fi in f:
                if isinstance(fi, (str, CausalNode)):
                    node = self[fi]
                    l.append(X[:, self.index(node)])
                else:
                    l.append(fi(X))
                    assert len(l[-1].shape) == 2, 'f must return a bidimensional tensor'

            X = torch.cat(l, 1)
            
        return X
    
    def _broadcast_procedure(self, n, x, *ds, ex_n=None, broadcast=None, sample=True):
        assert ex_n is not None
        
        if broadcast is None:
            broadcast = torch.zeros(n, dtype=bool, device=self.device)
        assert broadcast.shape == (n,) and broadcast.dtype == torch.bool
        
        # Detect missingness and broadcast if needed
        m = {
            node: torch.isnan(v).any(1)
            for node, v in x.items()
        }
                
        for node in self._nodes: # topological order
            m[node] = m.get(node, torch.ones((n,), dtype=bool, device=self.device))
            
            # If all its ancestors (including ex_noise) are fully specified, it's value is deterministic.
            # In that case, even if it is technically missing, we can recover its value.
            if sum(1 for _ in self.ancestors(node, all=True)) > 1: # not only the node itself
                deterministic = ~torch.stack([
                    m.get(a, torch.ones((n,), dtype=bool, device=self.device))
                    for a in self.ancestors(node, all=True)
                    if a is not node
                ], 1).any(1)

                m[node] = m[node] & ~deterministic

        broadcast |= torch.stack([v for v in m.values()], 1).any(1)
        idx = torch.arange(broadcast.size(0), device=self.device)
        nb_idx, b_idx = idx[~broadcast], idx[broadcast]

        idx = torch.cat([ nb_idx, b_idx.repeat(ex_n) ], 0)
        recover_idx = torch.argsort(torch.cat([ nb_idx, b_idx], 0))
        n = len(idx)
        x = { k: v[idx] for k, v in x.items() }
        m = { k: v[idx] for k, v in m.items() }
        
        ds = [
            { k: v[idx] for k, v in d.items() }
            for d in ds
        ]

        if sample:
            _, x = self.sample(n, interventions=x, return_all=True)
        
        return nb_idx, b_idx, recover_idx, n, x, m, *ds
    
    @requires_init
    def cond_exp(self, x, f=None, ex_n=100):
        """Compute the Conditional Expectation of f(V) given conditioning dict x."""
        x = self._preprocess_dict(x)
        n = max(v.size(0) for v in x.values())
        _, _, _, n, x, _ = self._broadcast_procedure(n, x, ex_n=ex_n, sample=False)

        original_x = { k: v for k, v in x.items() }

        # Sample the remaining ex_noise
        x.update({
            ex: ex.sample(n)
            for node in self.observable_nodes
            if node not in x
            for ex in node.ex_noise
        })

        # Sample U
        x.update({
            u: u.sample(n)
            for u in self.latent_nodes
        })

        x, d = self.sample(n, interventions=x, return_all=True)
        x = self._preprocess_fX(f, x)
        d = { k: v for k, v in d.items() if k not in original_x and k in self.observable_nodes }

        x = x.view(ex_n, -1, *([x.size(1)] if len(x.shape) > 1 else []))
        w = torch.softmax(self.loglk(original_x, cond=d).view(ex_n, -1, 1), 0)

        return (x * w).sum(0)

    @requires_init
    def loglk(
        self, x, target_node=None, interventions=None, cond=None, ex_n=100,
    ):
        r"""Compute the Log-Likelihood of a sample x.

        Args:
            x: samples to compute loglk.
                If x is a Tensor, it must include **all** observable nodes.
                If it is a dict {node: value}, it may contain 
                all or some observable nodes, and also some latent nodes.
            target_node (CausalNode): if not None, only compute loglk
                for that node. The result will be a single tensor 
                with just the node's loglk.
            interventions (dict): what interventions to apply, if any.
                Dict { node: value } with the constant value to apply 
                to each of the n samples. You can pass python scalar values,
                even booleans, or single dim tensors,
                and they will be transformed and broadcasted to all n samples.
                Note that intervened nodes won't sum their loglk values,
                as their probability is 1.
            cond (dict): conditioning term for the loglk.
            ex_n (int): how many samples to create per x sample
                if the total law of probability is required.

        Note that if any public node is not specified 
        (except for exogenous noise signals),
        this method will use the total law of probability
        to sample values for the non-specified nodes, 
        compute the corresponding loglk of the specified nodes
        and aggregate those with `dcg.utils.log_mean_exp_trick`.

        If there are missingness indicators, they will also be considered when appropriate.

        $$f(X) = \mathbb{E}_{\mathcal{V} - X}[ f(X \mid \mathcal{V} - X) ]$$
        """
        
        x, n = self._preprocess_x(x) # note that this copies x
        
        if cond is not None:
            x.update({ k: v for k, v in cond.items() })
            loglk = partial(self.loglk, target_node=target_node, interventions=interventions, ex_n=ex_n)
            
            return loglk(x) - loglk(cond)
        
        original_n = n

        if target_node is not None:
            target_node = self[target_node] # transform str to node
            assert not target_node.latent
            ancestors = list(self.ancestors(target_node, all=True))

        # Prepare interventions
        assert isinstance(interventions, dict) or interventions is None
        interventions = self._preprocess_dict(interventions or {}, n=n)

        for node, v in interventions.items():
            x[node] = v

        # Detect missingness and broadcast if needed
        nm_idx, m_idx, recover_idx, n, x, m = self._broadcast_procedure(n, x, ex_n=ex_n)

        # Now, compute loglk for each node
        
        # First, get theta
        if self.net is not None:
            theta = self.net(torch.cat([ x[node] for node in self.nodes ], 1))
        else:
            theta = None
        
        loglk = {}
        for node in self._nodes:
            # If we have a target_node, we're only concerned with its ancestors
            if target_node is not None and node not in ancestors:
                continue

            if (
                not node.latent and # only compute loglk of measured nodes
                node not in interventions and # if they're not intervened
                target_node in (None, node) # only the target_node, if any
            ):
                # Get the theta for this node, if there's any
                if theta is None or node not in self.theta_slices:
                    theta_node = None
                else:
                    theta_node = theta[:, self.theta_slices[node]]
                    
                loglk_node = node.loglk(
                    x[node], 
                    *(x.get(parent) for parent in node._parents),
                    theta=theta_node
                )

                loglk[node] = loglk_node = loglk_node * (1 - m[node].float())

                # Check that we didn't diverge
                if (
                    torch.isinf(loglk_node).any() or
                    torch.isnan(loglk_node).any()
                ):
                    warnings.warn(f'Divergence in loglk: {node}')

                # Check that its shape matches
                assert loglk_node.shape == (n,), (node, loglk_node.shape)

        # What to return?
        if target_node is not None:
            # We might not have computed the loglk of target_node
            # because it might also be intervened, 
            # in which case it's p(x) = 1 -> log p(x) = 0
            loglk = loglk.get(target_node, torch.zeros(n, device=self.device))
        else:
            loglk = sum(loglk.values()) # sums tensors together, not samples

        # Aggregate any missings
        loglk = torch.cat([
            loglk[:len(nm_idx)], # nm_idx
            log_mean_exp_trick(loglk[len(nm_idx):].view(ex_n, -1), dim=0) if m_idx.numel() else loglk[:0],
        ], 0)

        loglk = loglk[recover_idx]
        return loglk
    
    def nll(self, *args, **kwargs):
        return -self.loglk(*args, **kwargs)
    
    @requires_init
    def counterfactual(
        self, x, target_node=None, interventions=None, ex_n=100, f=None, agg=True
    ):
        """Counterfactual estimation.

        Computes counterfactual samples from x and interventions.
        If there are any missing public values, or a node is not ex_invertible,
        computes ex_n of them for each y sample.

        If agg is True, these extra samples are aggregated and
        the counterfactual expected value for each variable in the graph
        is returned in a single tensor.

        Otherwise, a tensor (ex_n, n, graph.dim) is returned with all samples.
        This can be used for the expectancy of functions 
        of the graph's counterfactual samples.

        Args:
            x: torch.Tensor or dict with observed samples.
            target_node: node to subselect once counterfactuals are sampled.
                Necessary to avoid sampling unnecessary nodes.
            interventions (dict): intervention values.
            ex_n (int): how many extra samples to take per x sample, if needed.
            f (func): function for which to compute the expectation. 
                Transforms the subsampled cf[target_node] (or cf if target_node is None).
            agg (bool): whether to aggregate extra samples.
                Note that the format of the return depends on this value.

        Returns:
            `x`: 
                tensor of shape (n, graph.dim), only if agg=True.
            `(x, w, nb_idx, b_idx, recover_idx)`:
                Only if agg=False.
                x is a tensor of shape (ex_n, n, dim), 
                    where dim is either graph.dim, target_node.dim or f(...).size(1).
                w is a tensor of shape (ex_n, n, 1)
                    or of no shape (1, if there weren't extra samples).
                    Contains the pre-softmax weights.
                nb_idx: indexer for non-broadcasted entries.
                b_idx: indexer for broadcasted entries.
                recover_idx: indexer to recover the original ordering once aggregation
                    has been taken care of.

        If you want the expectation of a function of the counterfactuals, 
        pass the desired function f and agg=True.
        
        Alternatively, pass agg=False and aggregate the results:
            ```python
            z = (f(x) * softmax(w)).sum(dim=0)
            ```
        """

        if target_node is not None:
            target_node = self[target_node]

        x, n = self._preprocess_x(x)
        original_n = n

        # Process interventions
        assert interventions is None or isinstance(interventions, dict)
        interventions = self._preprocess_dict(interventions or {}, n=n)
        
        # Any node not ex_invertible, all broadcast: 
        if any(not node.ex_invertible for node in self.nodes):
            broadcast = torch.ones(n, dtype=bool, device=self.device)
        else:
            broadcast = None
            
        nb_idx, b_idx, recover_idx, n, x, m, interventions = \
            self._broadcast_procedure(n, x, interventions, ex_n=ex_n, broadcast=broadcast)

        # Start processing
        
        # Abduction
        # Before this, we need to get theta (if there are external parameters)
        # Since we have values for every variable due to the previous procedure,
        # we can run it in a single pass.
        if self.net is not None:
            theta = self.net(torch.cat([ x[node] for node in self.nodes ], 1))
        else:
            theta = None
        
        ex_noise = {}

        # Any non-descendant of an intervened node will not change its value.
        # Hence, consider them as if they were intervened.
        # Also, if we have a target node, any non-ascendant of the target node
        # won't matter for the value of target_node.
        descendants = {
            desc
            for node in interventions.keys()
            for desc in self.descendants(node)
            if target_node is None or target_node in self.descendants(desc)
        }

        for node in self._nodes:
            if node not in descendants:
                interventions[node] = x[node]
                ex_noise[node] = tuple()

        # Next, abduct descendants
        # This will only affect ex_noises of non-missing nodes.
        # The rest of the missing values will be sampled normally
        # and taken into account later for importance sampling.
        w = torch.zeros(n, device=self.device)
        for node in self._nodes:
            assert node in x, node

            if node not in interventions: # if it's intervened, there's no use
                if theta is None or node not in self.theta_slices:
                    theta_node = None
                else:
                    theta_node = theta[:, self.theta_slices[node]]
                    
                if node.latent: 
                    # Latents have no ex_noise
                    # Use their sampled value as an intervention
                    interventions[node] = x[node]
                    ex_noise[node] = tuple() # no ex_noise
                else:
                    # Observable nodes do have ex_noise, so abduct
                    ex_noise[node] = node.abduct(
                        x[node], *(x[p] for p in node.parents), 
                        theta=theta_node
                    )

                assert isinstance(ex_noise[node], (tuple, list)), \
                    'abduct should return tuple of ex_noises (%s)' % node

                w += node.loglk(
                    x[node], 
                    *(x[p] for p in node.parents), 
                    theta=theta_node
                ) * (1 - m[node].float())
                
        # Sample in the intervened model:
        # Note that theta may differ now, but it is taken care of by the sample method.
        for node, t in ex_noise.items():
            for ex, v in zip(node.ex_noise, t):
                interventions[ex] = v
        _, x = self.sample(n, interventions=interventions, return_all=True)
                
        # Transform to tensor
        if target_node is None:
            x = self.dict_to_tensor(x)
        else:
            x = x[target_node]
            
        x = self._preprocess_fX(f, x)
            
        if agg:
            ex_n = n // original_n
            w = torch.softmax(w[len(nb_idx):].view(ex_n, -1), dim=0).unsqueeze(2)
            x = torch.cat([
                x[:len(nb_idx)], # nb needs no aggregation, as it hasn't been broadcasted
                (x[len(nb_idx):].view(ex_n, -1, x.size(1)) * w).sum(0)
            ], 0)

            return x[recover_idx]
        else:
            return x, w, nb_idx, b_idx, recover_idx