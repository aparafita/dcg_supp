import torch
from torch import nn

from .utils import isin

from functools import wraps


def requires_init(func):
    
    @wraps(func)
    def f(self, *args, **kwargs):
        assert not self.requires_initialization or self.initialized, \
            f'You must call warm_start before using {self}.'
        
        return func(self, *args, **kwargs)
    
    return f


class Module(nn.Module):
    """
    Base module with several utilities.
    
    For one, it has a `device` attribute,
    automatically updated whenever `cpu`/`cuda`/`to` methods are invoked.
    Also updates any submodules inside this module.
    Note that this will only work if its parent modules are also instances of this class.
    
    Caution: if your model contains an nn.ModuleList of torch_utils.Module,
    extend the _update_device method in your new class to set .device on all of its submodules.
    
    Additionally, contains a default `warm_start` method you can call after creating the network.
    If the `requires_initialization` class attribute is set to True,
    you must call this method before using the module, otherwise an AssertionError will be raised.
    This is used to force warm_start operations to be called on certain Modules.
    Override the `warm_start` method as needed. 
    However, remember to always start your warm_start overrides 
    with `super().warm_start(x)` and finish with `return self`. 
    Performing operations before calling `super()` could result in AssertionErrors, use with care.
    """
    
    requires_initialization = False # by default, we do not require warm_start
    
    def __init__(self):
        super().__init__()
        
        self._device = torch.device('cpu')
        self.register_buffer('_initialized', torch.tensor(False))
    
    @property
    def initialized(self):
        return self._initialized.item()
    
    @initialized.setter
    def initialized(self, value):
        assert isinstance(value, bool)
        self._initialized.data = torch.tensor(value).to(self.device)
    
    @property
    def device(self):        
        return self._device
    
    @device.setter
    def device(self, value):
        # We include the setter in a different function 
        # so it can be overriden easily if necessary.
        self._update_device(value)
                
    def _update_device(self, device):
        """Update device attribute for this module and all its sudmodules."""
        self._device = device
        
        for module in self.modules():
            if module is not self:
                # Note that if it's another Module, it will call this method recursively.
                # If not, it will just assign an attribute.
                module.device = device
                
    # Extensions to to/cpu/cuda in order to update device
    def to(self, device):
        self.device = device
            
        return super().to(device)
    
    def cpu(self):
        """Override .cpu so as to call .to method."""
        return self.to(torch.device('cpu'))
    
    def cuda(self):
        """Override .cuda so as to call .to method."""
        return self.to(torch.device('cuda', index=0))
    
    # warm_start operations
    def warm_start(self, *args, **kwargs):
        # Doesn't do anything, subclasses should define operations here if needed
        self.initialized = True
        
        return self
    
    @requires_init
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
    # When assigning modules to a property setter, PyTorch circumvents and overrides the property.
    # To avoid this, we use this code (https://github.com/pytorch/pytorch/issues/52664)
    def __setattr__(self, name, value):
        if name in dir(type(self)) and isinstance(getattr(type(self), name), property):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
    
    
class Sequential(Module):
    """`torch.nn.Sequential` extended to operate with `Module` devices
    and warm_start operation."""
    
    def __init__(self, *modules):
        super().__init__()
        
        self._modulelist = nn.Sequential(*modules)
        
    def forward(self, x):
        return self._modulelist(x)
    
    def __getitem__(self, k):
        return self._modulelist[k]
    
    def __iter__(self):
        return iter(self._modulelist)
    
    def __repr__(self):
        return repr(self._modulelist)
    
    def warm_start(self, x):
        super().warm_start(x)
        
        with torch.no_grad():
            for module in self._modulelist:
                if hasattr(module, 'warm_start'):
                    module.warm_start(x)

                x = module(x)
        
        return self
    
    
class Residual(Module):
    
    def __init__(self, f=None):
        """Args:
            f (function): function(x, res) that outputs the result of the residual connection.
                If None, the default operation (x + res) will be applied.
                Useful if the residual term and x don't have the same shape
                and a special operation is needed.
        """
        super().__init__()
        
        if f is None:
            f = lambda x, res: x + res
        
        self.f = f
        
    def forward(self, x, res):
        return self.f(x, res)
            
    
    
class ResidualSequential(Sequential):
    """Extension of `Sequential` to include residual connections."""
        
    def forward(self, x):
        res = 0
        for module in self:
            if isinstance(module, Residual):
                res = x = module(x, res)
            else:
                x = module(x)
            
        return x
        
    def warm_start(self, x):
        super(Sequential, self).warm_start(x)
        
        res = 0
        for i, module in enumerate(self):
            if hasattr(module, 'warm_start'):
                module.warm_start(x)
            
            if isinstance(module, Residual):
                res = x = module(x, res)
            else:
                x = module(x)
            
        return self


class Lambda(Module):
    """Module used to run a custom lambda function."""
    
    def __init__(self, f):
        super().__init__()
        
        self.f = f
    
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)
    

class Normalizer(Module):
    """Normalization layer, designed for the beginning of a Network to normalize the input.
    
    Does not learn over the course of training, as it assumes a stable input distribution.
    As such, must be initialized before ever being used (otherwise, it raises an AssertionError)
    by calling warm_start with a tensor from the input distribution.
    """
    
    requires_initialization = True
    
    def __init__(self, num_features, affine=True, eps=1e-05):
        """
        Args:
            num_features (int): number of input dimensions.
            affine (bool): whether to add two learnable parameters, 
                loc and scale, after the normalization.
            eps (float): small number to add to scale = exp(logscale) + eps.
        """
        super().__init__()
        
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        
        self.register_buffer('distr_mean', torch.randn(num_features))
        self.register_buffer('distr_scale', torch.ones(num_features))
        
        if self.affine:
            self.bias = nn.Parameter(torch.randn(num_features))
            self.weight = nn.Parameter(torch.randn(num_features))
            
    def warm_start(self, x):
        super().warm_start(x)
        
        self.distr_mean.data = x.mean(0)
        self.distr_scale.data = x.std(0)
        
        return self
        
    def forward(self, x):        
        x = (x - self.distr_mean) / self.distr_scale
        if self.affine:
            x = x * (torch.exp(self.weight) + self.eps) + self.bias
            
        return x
    
    
class NoiseAugmentation(Module):
    """Identity flow, that adds 0-centered normal noise on forward-transform only during training."""
    
    def __init__(self, *args, sigma=.1, **kwargs):
        assert sigma > 0
        
        super().__init__(*args, **kwargs)
        
        self.register_buffer('sigma', torch.tensor(sigma))
        
    def forward(self, x):
        # Transform x into u. Used for training.
        if self.training:
            x = x + torch.randn_like(x) * self.sigma
            
        return x
    

class MaskedLinear(nn.Linear):
    """Extend `torch.nn.Linear` to use a boolean mask on its weights."""

    def __init__(self, in_features, out_features, bias=True, mask=None):
        """
        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sampl.
            bias (bool): If set to False, 
                the layer will not learn an additive bias. 
            mask (torch.Tensor): boolean mask to apply to the weights.
                Tensor of shape (out_features, in_features).
                If None, defaults to an unmasked Linear layer.
        """
        super().__init__(in_features, out_features, bias)
        
        if mask is None:
            mask = torch.ones(out_features, in_features, dtype=bool)
        else:
            mask = mask.bool()

        assert mask.shape == (out_features, in_features)
        self.register_buffer('_mask', mask)
        
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, value):
        self._mask.data = mask.to(self._mask.device)

    def forward(self, input):
        """Extend forward to include the buffered mask."""
        return nn.functional.linear(input, self.mask * self.weight, self.bias)
    

def intersections(first_layer, K):
    """Return all set intersections of up to K sets.
    
    Args:
        first_layer (dict): dict {i: s} where i is the output dim index
            and s is a (possibly empty) set containing all input dimensions j in s
            that output dimension i depends on. Any empty sets s will be discarded.
        K (int): maximum number of sets each intersection should consider.
            If None, K = len(first_layer)
            
    Note that for high number of output dimensions this method could take a while
    and/or return a huge number of sets. Preferably test how far you can go with K
    before asking for K=None. 
    """
    prev = first_layer = [ ([i], s) for i, s in first_layer.items() if s ]
    
    K = len(prev) if K is None else min(K, len(prev))
        
    all_sets = { tuple(sorted(s)) for _, s in first_layer if s }
    for k in range(1, K + 1):
        current = []

        for I, comb in prev:
            i = max(I)
            for (j,), original_comb in first_layer:
                if j > i:
                    comb2 = comb.intersection(original_comb)
                    if comb2:
                        current.append((I + [j], comb2))

        # Drop duplicates in current
        # Since all combs are added in order,
        # and we want to take the last I for every duplicate pair,
        # as this will reduce the number of intersections to perform,
        # we can just add constantly to a dict, that will overwrite the set keys
        # with the latest I.
        current = { tuple(sorted(s)): I for I, s in current }

        # Recover the original format
        current = sorted((
            (I, set(s))
            for s, I in current.items()
        ), key=lambda pair: pair[0])

        if not current: break
        all_sets = all_sets.union({ tuple(sorted(s)) for _, s in current if s })
        prev = current
        
    return sorted(all_sets)
      
    
class DAGMaskedNet(Module):
    
    def __init__(
        self, 
        input_dim, n_subnetworks, cond_dim=0,
        K=2, output_deps=None, output_m=1, 
        hidden_m=1, n_layers=None, residual=1,
        use_bn=True, activation=nn.ReLU, dropout=.1, 
        init=None
    ):
        """
        Args:
            input_dim (int): number of dimensions in the input.
            n_subnetworks (int): number of networks to combine.
            cond_dim (int): number of dimensions for the conditioner input (can be 0).
            K (int): number of elements per intersection to consider.
                If None, the maximum number of intersections will be used.
            output_deps (dict): dict { i::int: s::set} that relates each subnetwork i 
                with its input dependencies s expressed as sets of integers.
                If list, each position i reflects value dict[i]. 
                If None, autoregressive dependencies expected (i depends on 1..i-1).
            output_m (int, list): number of output dimensions per subnetwork. 
                If int, the same number will be used for all subnetworks.
                If list, each value represents the number corresponding to each subnetwork.
            hidden_m (int, list): multiplier for hidden dimensions. 
                If int, the same multiplier will be used for each hidden layer.
                If list, each layer will have its own multiplier.
            n_layers (int): number of MaskedLinear dimensions to use in the whole network.
                If None, hidden_m must be a tuple/list containing the multiplier for each hidden layer,
                and n_layers it is inferred from hidden_m (n_layers = len(hidden_m) + 1).
            residual (int): how many MaskedLinear steps between each residual connection. 
                If 0, no residual connections used.
            use_bn (bool): whether to use BatchNormalization in between blocks.
            activation (nn.Module): activation to use after MaskedLinear layers (except the last one).
            dropout (float): factor of Dropout to use between blocks. If 0, no Dropout used.
        
        NOTE: This network MUST be warm started (call warm_start(X)) before use.
        """
        assert isinstance(hidden_m, (tuple, list)) or n_layers > 0
        assert residual >= 0
        
        super().__init__()
        
        self.input_dim = input_dim
        self.n_subnetworks = n_subnetworks
        self.cond_dim = cond_dim
        
        if output_deps is None:
            # Autoregressive
            assert input_dim == n_subnetworks
            
            output_deps = { 
                i: set(range(i)) 
                for i in range(input_dim) 
            }
        elif isinstance(output_deps, (list, tuple)):
            output_deps = { 
                i: set(s)
                for i, s in enumerate(output_deps) if not cond_dim and s 
            }
            
        if cond_dim:
            for s in output_deps.values():
                s.add(-1)
            
        assert (
            set(output_deps.keys()) <= set(range(self.n_subnetworks)) and
            all(isinstance(s, set) for s in output_deps.values()) and
            set(x for s in output_deps.values() for x in s) <= set(range(-1 if cond_dim else 0, input_dim))
        )
        
        # Remove any keys that have an empty s
        for i, s in list(output_deps.items()): # list because we're gonna modify the dict
            if not s:
                output_deps.pop(i)
                
        self.output_deps = output_deps
                
        # Define two sets to quickly identify if an input dimension has been used
        self.used_input_dims = { x for s in output_deps.values() for x in s if x >= 0 }
        assert self.used_input_dims, 'All independent output_dims not supported.'
        self.unused_input_dims = set(range(input_dim)) - self.used_input_dims
        # At least one dimension should not be an input
        assert self.unused_input_dims
        
        # Define two sets to quickly identify if a subnetwork has no dependencies
        self.dep_subnetworks = set(output_deps.keys())
        self.indep_subnetworks = set(range(n_subnetworks)) - self.dep_subnetworks
        
        # Resolve hidden_m
        if isinstance(hidden_m, int):
            assert n_layers is not None
            hidden_m = [hidden_m] * (n_layers - 1)
        else:
            if n_layers is None:
                n_layers = len(hidden_m) + 1
                
            assert isinstance(hidden_m, (tuple, list)) and \
                   len(hidden_m) == n_layers - 1 and \
                   all(isinstance(m, int) and m > 0 for m in hidden_m)            
        
        self.hidden_m = hidden_m
        
        # Resolve output_m
        if isinstance(output_m, int):
            output_m = [output_m] * n_subnetworks
        else:
            assert isinstance(output_m, (list, int)) and \
                   len(output_m) == n_subnetworks and \
                   all(m > 0 for m in output_m)
            
        self.output_m = output_m
        self.output_dim = sum(self.output_m)
        
        # Define masks
        sets = intersections(self.output_deps, K=K)
        
        def mask_from_combs(upper, lower):
            upper = [set(x) for x in upper]
            lower = [set(x) for x in lower]

            mask = torch.zeros(len(upper), len(lower), dtype=bool)
            for i, s in enumerate(upper):
                for j, s2 in enumerate(lower):
                    mask[i, j] = s2 <= s

            return mask

        masks = list(reversed(
            [mask_from_combs([s for s in self.output_deps.values()], sets)] +
            [mask_from_combs(sets, sets) for _ in range(n_layers - 2)] +
            [mask_from_combs(sets, ([(-1,)] if self.cond_dim else []) + [(i,) for i in sorted(self.used_input_dims)])]
        ))
        
        if cond_dim:
            masks[0] = torch.cat([
                masks[0][:, [0]].repeat(1, cond_dim),
                masks[0][:, 1:]
            ], 1)
        
        # Hidden layers must be repeated hidden_m times
        for i, (mi, mo) in enumerate(
            zip([1] + self.hidden_m, self.hidden_m + [1])
        ):
            masks[i] = masks[i].repeat(mo, mi)
            
        # Output mask must be repeated output_m times for each dimension.
        # Note that independent subnetworks do not appear here.
        masks[-1] = torch.cat([
            masks[-1][[i]].repeat(self.output_m[j], 1)
            for i, j in enumerate(self.dep_subnetworks)
        ], 0)
        
        # Create the network
        i = 0 # used to identify residual connections
        net = [] # we'll add layers to the Sequential net
        
        residuals = []
        for n_layer, mask in enumerate(masks):
            if use_bn:
                net.append(nn.BatchNorm1d(mask.size(1), affine=True))
                i += 1
                
                if activation is not None:
                    net.append(activation())
                    i += 1
                    
            if n_layer > 0: 
                residuals.append(i)
                
            if n_layer > 0 and dropout > 0:
                net.append(nn.Dropout(dropout))
                i += 1
            
            net.append(MaskedLinear(*reversed(mask.shape), bias=True, mask=mask))
            i += 1
                    
        self.net = nn.Sequential(*net)
        
        self.use_bn = use_bn
        self.residual = bool(residual)
        self.residuals = residuals[:-1:residual] if self.residual else None
                    
        # Add bias tensors for those subnetworks that don't depend on the input
        if self.indep_subnetworks:
            self.indep_params = nn.Parameter(torch.randn(
                sum(m for i, m in enumerate(self.output_m) if i in self.indep_subnetworks)
            ))
            
            self.register_buffer(
                'indep_params_mask', 
                torch.Tensor(sum(
                    (
                        [i] * self.output_m[i]
                        for i in self.indep_subnetworks
                    ), []
                ))
            )
            
        self.register_buffer(
            'dep_params_mask',
            torch.Tensor(sum(
                (
                    [i] * self.output_m[i]
                    for i in self.dep_subnetworks
                ), []
            ))
        )
            
        self.register_buffer(
            'output_mask', 
            torch.Tensor(sum(([i] * m for i, m in enumerate(self.output_m)), []))
        )
        
        if init is not None:            
            for i in range(self.n_subnetworks):
                if i in self.dep_subnetworks:
                    self.net[-1].bias.data[self.dep_params_mask == i] = \
                        init[self.output_mask == i]
                else:
                    self.indep_params.data[self.indep_params_mask == i] = \
                        init[self.output_mask == i]
                    
            self.net[-1].weight.data = torch.randn_like(self.net[-1].weight.data) * 1e-3
        
                    
    def warm_start(self, x, cond=None):
        if self.cond_dim:
            assert cond is not None and cond.size(1) == self.cond_dim
            super().warm_start(torch.cat([cond, x], 1))
        else:
            super().warm_start(x)
        
        x = x[:, sorted(self.used_input_dims)]
        if self.cond_dim:
            x = torch.cat([cond, x], 1)

        if not self.residual:
            with torch.no_grad():
                for layer in self.net:
                    if hasattr(layer, 'warm_start'):
                        layer.warm_start(x)

                    x = layer(x)
        else: 
            with torch.no_grad():
                res = 0
                j = 0
                for i, layer in enumerate(self.net):
                    if hasattr(layer, 'warm_start'):
                        layer.warm_start(x)

                    if j < len(self.residuals) and i == self.residuals[j]:
                        res, x = x, layer(x) + res
                        j += 1
                    else:
                        x = layer(x)
        
        return self
    
    
    def forward(self, x, cond=None):
        x = x[:, sorted(self.used_input_dims)]
        if self.cond_dim:
            assert cond is not None and cond.size(1) == self.cond_dim
            x = torch.cat([cond, x], 1)
        
        if not self.residual:
            x = self.net(x)
        else:
            res = 0
            j = 0
            for i, layer in enumerate(self.net):
                if j < len(self.residuals) and i == self.residuals[j]:
                    res, x = x, layer(x) + res
                    j += 1
                else:
                    x = layer(x)
                    
        # Now, join the independent output biases with the rest of the tensor
        res = torch.zeros(x.size(0), self.output_dim, device=x.device)
        filtered_output_mask = self.output_mask[isin(self.output_mask, self.dep_subnetworks)]
        for i in range(self.n_subnetworks):
            if i in self.dep_subnetworks:
                res[:, self.output_mask == i] = x[:, filtered_output_mask == i]
            else:
                res[:, self.output_mask == i] = \
                    self.indep_params[self.indep_params_mask == i].unsqueeze(0)
                
        return res
    
    
class AdjacencyMaskedNet(Module):
    """Module to define independencies input-output through an adjacency matrix.
    
    It allows to define a network with arbitrary architecture 
    that still satisfies the given independencies.
    """
    
    def __init__(self, A, net_f=None, init=None):
        assert net_f is not None
        
        super().__init__()
        
        self.input_dim, self.output_dim = A.shape
        self.register_buffer('A', A.bool())
        
        # Remove any duplicate columns from A
        from collections import defaultdict
        d = defaultdict(set)
        
        for j in range(A.size(1)):
            code = tuple( bool(x) for x in A[:, j] )
            d[code].add(j)
            
        dk = sorted(d.keys())
            
        A_ = torch.Tensor(dk).bool().t()
        self.register_buffer('A_', A_)
        
        idx = torch.zeros(A.size(1), A_.size(1), dtype=bool)
        for j, code in enumerate(dk):
            for i in d[code]:
                idx[i, j] = True
        self.register_buffer('idx', idx)
        
        self.net = net_f(*A.shape, init=init)
        
    def forward(self, x, subidx=None):
        assert x.shape == (len(x), self.input_dim)
        
        idx = self.idx.to(self.device)
        A_ = self.A_
        
        if subidx is not None:
            assert subidx.shape == (self.A.size(1),)
            
            any_ = idx[subidx].any(0)
            idx, A_ = idx[:, any_] & subidx.unsqueeze(1), A_[:, any_]
        
        (n, i), o, o_ = x.shape, self.A.size(1), A_.size(1)
    
        # Multiply by A_ in dimension 2 to have as many masked inputs as needed
        x = x.unsqueeze(2) * A_.unsqueeze(0)
        x = x.permute(0, 2, 1).reshape(-1, i)
        
        # Run the input through the network
        assert x.shape == (n * o_, i)
        x = self.net(x)
        
        # And extract the required output dimensions
        assert x.shape == (n * o_, o)
        x = x.view(n, o_, o).permute(0, 2, 1)[:, idx]
        
        return x
    
    def warm_start(self, *args, **kwargs):
        if hasattr(self.net, 'warm_start'):
            self.net.warm_start(*args, **kwargs)
            
        return super().warm_start(*args, **kwargs)
    

class MultiHeadNet(Module):
    
    def __init__(
        self, 
        input_dim, head_slices=None, 
        base_f=None, head_f=None
    ):
        """Create a MultiHead network.
        
        Parameters:
            - input_dim (int): dimension of the input conditioning tensor.
            - head_slices (list): slices of the dimensions to use for multiheading.
            - base_f (function): function(input_dim) that returns the base network to use.
            - head_f (function): function(base_net) that returns the network to use for each head after base_net.
        """
        super().__init__()
        
        assert head_slices, 'head_slices must be non-empty'
        assert all(f is not None for f in (base_f, head_f)), \
            'Base and Head functions must be provided'
        
        self.register_buffer('slice_mask', torch.zeros(input_dim, dtype=bool))
        slice_indices = []
        
        n_heads = 1
        indices = list(range(input_dim))
        for sl in head_slices:
            self.slice_mask[sl] = True
            
            sl = indices[sl]
            assert sl, 'Slice led to empty list'
            n_heads *= max(2, len(sl))

            slice_indices.append(sl)

        self.base = base_f(input_dim - self.slice_mask.int().sum().item())

        all_unique = lambda l: len(set(l)) == len(l)
        assert all_unique(sum(slice_indices, [])), \
            'head_slices must contain unique indices'

        self.heads = nn.Sequential(*(
            head_f(self.base)
            for _ in range(n_heads)
        ))
        
        head_i = 0
        def recursive(combs, slice_indices):
            if slice_indices:
                sl = slice_indices[0]
                d = max(2, len(sl))
                
                for i in range(d):
                    subcombs = combs[:, i::d]
                    
                    if len(sl) == 1:
                        subcombs[sl] = bool(i)
                    else:
                        subcombs[sl] = (torch.arange(d) == i).unsqueeze(1)
                        
                    recursive(subcombs, slice_indices[1:])
                    
            return combs[self.slice_mask]

        self.register_buffer(
            'combs', 
            recursive(
                torch.zeros(input_dim, len(self.heads), dtype=bool), 
                slice_indices
            )
        )
        
        self.requires_initialization = (
            getattr(self.base, 'requires_initialization', False) or
            any(getattr(head, 'requires_initialization', False) for head in self.heads)
        )
        
    def forward(self, x):
        comb = x[:, self.slice_mask] > .5
        x = x[:, ~self.slice_mask]
        
        x = self.base(x)
        
        comb = (comb.unsqueeze(-1) == self.combs.unsqueeze(0)).all(1).float()
        x = (torch.stack([h(x) for h in self.heads], -1) * comb.unsqueeze(1)).sum(-1)
        
        return x
    
    def warm_start(self, x):
        super().warm_start(x)
        
        training = self.training
        self.eval()

        with torch.no_grad():
            x = x[:, ~self.slice_mask]
            if hasattr(self.base, 'warm_start'):
                self.base.warm_start(x)

            x = self.base(x)
            
            for head in self.heads:
                if hasattr(head, 'warm_start'):
                    head.warm_start(x)
        
        self.train(training)
        return self
    

from functools import partial

class SharedModule(nn.Module):
    """Used to share across different Modules.
    
    In order to use it, inherit this class 
    and override the forward method with your module implementation.
    Save the instanced module in a single other module affected by the optimizer.
    
    Then call the share method with any *args and **kwargs,
    that will be passed to the resulting function when called.
    The result of this call can be stored in submodules that "share" this module.
    
    Look at the `examples` section on the Github repository for an example of this technique.
    """
    # TODO: Add an example for this
        
    def share(self, *args, **kwargs):
        return partial(self.__call__, *args, **kwargs)