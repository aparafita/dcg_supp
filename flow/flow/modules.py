"""
Miscellaneous Flows.
"""

import numpy as np
import torch
from torch import nn

from .flow import Flow
from .utils import softplus, softplus_inv, logsigmoid, cache, cache_only_in_eval

from functools import partial


class Normalizer(Flow):
    """Normalizes a signal from a constant distribution.

    Provides the warm_start method to learn the constant parameters
    with which we will transform the distribution to have location 0 and variance 1.

    Note that it requires to be initialized with `warm_start`.
    """
    
    requires_initialization = True

    def __init__(self, eps=1e-6, **kwargs):
        """
        Args:
            eps (float): lower-bound for the weight tensor.
        """
        super().__init__(**kwargs)
        
        self.register_buffer('_log_weight', torch.zeros(self.dim))
        self.register_buffer('bias', torch.zeros(self.dim))
        self.register_buffer('eps', torch.tensor(eps))

    def _warm_start(self, x, **kwargs):
        self._log_weight.data = torch.log(x.std(0) + self.eps)
        self.bias.data = x.mean(0)
        
        super()._warm_start(x, **kwargs)
        
    @property
    def weight(self):
        return torch.exp(self._log_weight)
    
    def _log_abs_det(self, n):
        return self._log_weight.sum().unsqueeze(0).repeat(n)
        
    def _transform(self, x, *theta, log_abs_det=False, **kwargs):
        u = (x - self.bias) / self.weight
        
        if log_abs_det:
            return u, -self._log_weight.sum().unsqueeze(0).repeat(u.size(0))
        else:
            return u
    
    def _invert(self, u, *theta, log_abs_det=False, **kwargs):
        x = u * self.weight + self.bias
        
        if log_abs_det:
            return x, self._log_weight.sum().unsqueeze(0).repeat(x.size(0))
        else:
            return x
        
        
class NoiseAugmentation(Flow):
    """Identity flow, that adds 0-centered normal noise on forward-transform only during training."""
    
    def __init__(self, *args, sigma=.1, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.register_buffer('sigma', torch.tensor(sigma))
        
    def _transform(self, x, log_abs_det=False, **kwargs): 
        # Transform x into u. Used for training.
        if self.training:
            x = x + torch.randn_like(x) * self.sigma
        
        if log_abs_det:
            return x, torch.zeros_like(x[:, 0])
        else:
            return x

    def _invert(self, u, log_abs_det=False, **kwargs): 
        # Transform u into x. Used for sampling.
        if log_abs_det:
            return u, torch.zeros_like(u[:, 0])
        else:
            return u
        
        
@cache_only_in_eval
class LowerTriangular(Flow):
    
    @staticmethod
    def create_mask(N, mask=None):
        if mask is None:
            # Autoregressive
            mask = [ (i, j) for i in range(N) for j in range(i + 1, N) ]
        elif isinstance(mask, (list, tuple, set)):
            # Make sure it's in topological order, formed of (i, j) pairs with i < j
            assert all(
                isinstance(t, (list, tuple)) and 
                len(t) == 2 and 
                t[0] < t[1] 
                
                for t in mask
            )
        elif isinstance(mask, torch.Tensor):
            # Convert to a bool mask and 
            # make sure it's lower triangular with an all-True diagonal
            mask = mask.bool()
            assert torch.diagonal(mask).all().item() and \
                (~mask[torch.triu_indices(N, N, offset=1)]).all().item()
            
            return mask
        else:
            raise ValueError(type(mask))
            
        M = torch.eye(N, N, dtype=bool)
        for i, j in mask:
            M[j, i] = True # inverse order!
        
        return M
    
    def __init__(self, bias=True, mask=None, eps=1e-3, **kwargs):
        super().__init__(**kwargs)
        
        if bias:
            self.bias = nn.Parameter(torch.randn(self.dim))
        else:
            self.register_buffer('bias', torch.tensor(0.))
            
        self.register_buffer('mask', self.create_mask(self.dim, mask))
        self.register_buffer('eps', torch.tensor(eps))
        
        # The matrix will be initialized to something close to the identity
        eye = torch.eye(self.dim)
        A = eye * softplus_inv(1. - self.eps) + (1 - eye) * torch.randn_like(eye) * self.eps
        
        self._h = nn.Parameter(A[self.mask])
        
        # Create a mask for the diagonal terms
        self.register_buffer('diag_mask', torch.eye(self.dim, dtype=bool)[self.mask])
        
    @cache
    def A(self):
        A = torch.zeros(self.dim, self.dim, device=self.device)
        eye = torch.eye(self.dim, dtype=bool, device=self.device)
        
        A[eye] = torch.nn.functional.softplus(self._h[self.diag_mask]) + self.eps
        A[self.mask & ~eye] = self._h[~self.diag_mask]
        
        return A
    
    @cache
    def A_inv(self):
        A = self.A
        inv, _ = torch.triangular_solve(
            torch.eye(self.dim, device=self.device), 
            A, upper=False
        )
        
        return inv
        
    def _transform(self, x, log_abs_det=False, **kwargs): 
        # Transform x into u. Used for training.
        A = self.A
        u = x @ A.t() + self.bias
        
        if log_abs_det:
            return u, torch.log(torch.diagonal(A)).sum().unsqueeze(0)
        else:
            return u

    def _invert(self, u, log_abs_det=False, **kwargs): 
        # Transform u into x. Used for sampling.
        A_inv = self.A_inv
        x = (u - self.bias) @ A_inv.t()
        
        if log_abs_det:
            return x, torch.log(torch.diagonal(A_inv)).sum().unsqueeze(0)
        else:
            return x


@cache_only_in_eval
class Linear(Flow):
    """Invertible Linear Flow based on LU decomposition."""
    
    def __init__(self, *args, eps=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        
        d = self.dim
        
        self.bias = nn.Parameter(torch.randn(d) * .1)
        
        self._L_flat = nn.Parameter(torch.randn(d * (d - 1) // 2) * .1)
        
        U_flat = torch.randn(d * (d + 1) // 2) * .1
        U_flat[[i == j for i in range(d) for j in range(i, d)]] += 1
        self._U_flat = nn.Parameter(U_flat)
        
        self.register_buffer('eps', torch.tensor(eps))
        
        self.tril = torch.tril_indices(d, d, offset=-1, device=self.device)
        self.triu = torch.triu_indices(d, d, device=self.device)
        
    @cache
    def L(self):
        d = self.dim
        
        L = torch.eye(d, device=self.device)
        L[self.tril[0], self.tril[1]] = self._L_flat
        
        return L
        
    @cache
    def U(self):
        d = self.dim
        
        U = torch.zeros(d, d, device=self.device)
        U[self.triu[0], self.triu[1]] = self._U_flat
        
        return U + torch.diag(torch.sign(torch.diag(U)) * self.eps)
    
    @cache
    def A(self):
        return self.L @ self.U
        
    @cache
    def inv(self):
        inv, _ = torch.triangular_solve(
            torch.eye(self.dim, device=self.device), self.L, upper=False, 
            unitriangular=True
        )
        
        inv, _ = torch.triangular_solve(
            inv, self.U, upper=True, unitriangular=False
        )
        
        return inv
    
    def _log_abs_det(self, x, L, U):
        return torch.log(torch.abs(torch.diag(U))).sum().repeat(x.size(0))
    
    def _transform(self, x, log_abs_det=False, **kwargs): 
        # Transforms x into u. Used for training.
        L, U = self.L, self.U
            
        u = (L @ (U @ x.t())).t() + self.bias
        
        if log_abs_det:
            return u, self._log_abs_det(x, L, U)
        else:
            return u
            
    def _invert(self, u, log_abs_det=False, **kwargs): 
        # Transforms u into x. Used for sampling.
        L, U, inv = self.L, self.U, self.inv
        x = (inv @ (u - self.bias).t()).t()
        
        if log_abs_det:
            return x, -self._log_abs_det(x, L, U)
        else:
            return x
        

class Scaler(Flow):
    """Scaler Flow. Transforms by multiplying by a positive scale dim-wise."""

    def __init__(self, eps=1e-6, **kwargs):
        """
        Args:
            eps (float): lower-bound to the scale parameter.
        """
        super().__init__(**kwargs)

        assert eps > 0
        self.eps = eps
        
        self.logscale = nn.Parameter(torch.randn(1, self.dim))

    def _log_abs_det(self, x):
        return self.logscale.sum(1)
    
    @property
    def scale(self):
        return torch.exp(self.logscale) + self.eps

    # Override methods
    def _transform(self, x, log_abs_det=False, **kwargs):
        u = x * self.scale

        if log_abs_det:
            return u, self._log_abs_det(x)
        else:
            return u

    def _invert(self, u, log_abs_det=False, **kwargs):
        x = u / self.scale

        if log_abs_det:
            return x, -self._log_abs_det(x)
        else:
            return x 

    def _warm_start(self, x, **kwargs):
        self.logscale.data = torch.log(x.std(0, keepdim=True))
        
        super()._warm_start(x, **kwargs)
        
        
class OddRoot(Flow):
    """Odd Root Flow (|x|^(1/a) * sign(x), with a being an odd positive integer)"""
    
    def __init__(self, a=None, eps=1e-3, **kwargs):
        assert isinstance(a, int) and a % 2 and a > 0
        
        super().__init__(**kwargs)
        
        self.register_buffer('power', torch.tensor(1 / a))
        self.register_buffer('eps', torch.tensor(eps))
        
    def _log_abs_det(self, x):
        return (
            torch.log(self.power) + 
            (self.power - 1) * torch.log(x.abs() + self.eps)
        ).sum(1)
    
    # Override methods
    def _transform(self, x, log_abs_det=False, **kwargs):
        u = torch.pow(x.abs(), self.power) * torch.sign(x)
        
        if log_abs_det:
            return u, self._log_abs_det(x)
        else:
            return u
        
    def _invert(self, u, log_abs_det=False, **kwargs):
        x = torch.pow(u.abs(), 1 / self.power) * torch.sign(u)
        
        if log_abs_det:
            return x, -self._log_abs_det(x)
        else:
            return x
        
Cbrt = partial(OddRoot, a=3)


class Exponential(Flow):
    """Exponential Flow"""

    def __init__(self, eps=1e-6, **kwargs):
        """
        Args:
            eps (float): lower-bound to the scale parameter.
        """
        super().__init__(**kwargs)

        assert eps > 0
        self.eps = eps

    def _log_abs_det(self, x):
        return x.sum(1)

    # Override methods
    def _transform(self, x, log_abs_det=False, **kwargs):
        u = torch.exp(x) + self.eps

        if log_abs_det:
            return u, self._log_abs_det(x)
        else:
            return u

    def _invert(self, u, log_abs_det=False, **kwargs):
        x = torch.log(u.clamp(self.eps))

        if log_abs_det:
            return x, -self._log_abs_det(x)
        else:
            return x 


class Sigmoid(Flow):
    """Sigmoid Flow."""

    def __init__(self, alpha=1., eps=1e-3, **kwargs):
        r"""
        Args:
            alpha (float): alpha parameter for the sigmoid function: 
                \(s(x, \alpha) = \frac{1}{1 + e^{-\alpha x}}\).
                Must be bigger than 0.
            eps (float): transformed values will be clamped to (eps, 1 - eps) 
                on both _transform and _invert.
        """
        super().__init__(**kwargs)

        self.alpha = alpha
        self.eps = eps

    def _log_abs_det(self, x):
        """Return log|det J_T|, where T: x -> u."""
        return (
            np.log(self.alpha) + 
            2 * logsigmoid(x, alpha=self.alpha) +
            -self.alpha * x
        ).sum(dim=1)

    # Override methods
    def _transform(self, x, log_abs_det=False, **kwargs):
        u = torch.sigmoid(self.alpha * x)
        u = u.clamp(self.eps, 1 - self.eps)

        if log_abs_det:
            return u, self._log_abs_det(x)
        else:
            return u

    def _invert(self, u, log_abs_det=False, **kwargs):
        u = u.clamp(self.eps, 1 - self.eps)
        x = -torch.log(1 / self.alpha / u - 1)

        if log_abs_det:
            return x, -self._log_abs_det(x)
        else:
            return x


class Softplus(Flow):
    """Softplus Flow."""

    def __init__(self, threshold=20., eps=1e-6, **kwargs):
        """
        Args:
            threshold (float): values above this revert to a linear function. 
                Default: 20.
            eps (float): lower-bound to the softplus output.
        """
        super().__init__(**kwargs)

        assert threshold > 0 and eps > 0
        self.threshold = threshold
        self.eps = eps

    def _log_abs_det(self, x):
        return logsigmoid(x).sum(dim=1)

    # Override methods
    def _transform(self, x, log_abs_det=False, **kwargs):
        u = softplus(x, threshold=self.threshold, eps=self.eps)

        if log_abs_det:
            return u, self._log_abs_det(x)
        else:
            return u

    def _invert(self, u, log_abs_det=False, **kwargs):
        x = softplus_inv(u, threshold=self.threshold, eps=self.eps)

        if log_abs_det:
            return x, -self._log_abs_det(x)
        else:
            return x 


class LogSigmoid(Flow):
    """LogSigmoid Flow, defined for numerical stability."""

    def __init__(self, alpha=1., **kwargs):
        """
        Args:
            alpha (float): alpha parameter used by the `Sigmoid`.
        """
        super().__init__(**kwargs)

        self.alpha = alpha

    def _log_abs_det(self, x):
        """Return log|det J_T|, where T: x -> u."""

        return logsigmoid(-self.alpha * x).sum(dim=1) + np.log(self.alpha)

    # Override methods
    def _transform(self, x, log_abs_det=False, **kwargs):
        u = logsigmoid(x, alpha=self.alpha)

        if log_abs_det:
            return u, self._log_abs_det(x)
        else:
            return u

    def _invert(self, u, log_abs_det=False, **kwargs):
        x = -softplus_inv(-u) / self.alpha

        if log_abs_det:
            return x, -self._log_abs_det(x)
        else:
            return x
    

class LeakyReLU(Flow):
    """LeakyReLU Flow."""

    def __init__(self, negative_slope=0.01, **kwargs):
        """
        Args:
            negative_slope (float): slope used for those x < 0,
        """
        super().__init__(**kwargs)

        self.negative_slope = negative_slope

    def _log_abs_det(self, x):
        return torch.where(
            x >= 0, 
            torch.zeros_like(x), 
            torch.ones_like(x) * np.log(self.negative_slope)
        ).sum(dim=1)


    # Override methods
    def _transform(self, x, log_abs_det=False, **kwargs):
        u = torch.where(x >= 0, x, x * self.negative_slope)

        if log_abs_det:
            return u, self._log_abs_det(x)
        else:
            return u

    def _invert(self, u, log_abs_det=False, **kwargs):
        x = torch.where(u >= 0, u, u / self.negative_slope)

        if log_abs_det:
            return x, -self._log_abs_det(x)
        else:
            return x


class BatchNorm(Flow):
    """Perform BatchNormalization as a Flow class.

    If not affine, just learns batch statistics to normalize the input.
    """

    @property
    def affine(self):
        return self._affine.item()
    

    def __init__(self, affine=True, momentum=.1, eps=1e-6, **kwargs):
        """
        Args:
            affine (bool): whether to learn parameters loc/scale.
            momentum (float): value used for the moving average
                of batch statistics. Must be between 0 and 1.
            eps (float): lower-bound for the scale tensor.
        """
        super().__init__(**kwargs)

        assert 0 <= momentum and momentum <= 1

        self.register_buffer('eps', torch.tensor(eps))
        self.register_buffer('momentum', torch.tensor(momentum))

        self.register_buffer('batch_loc', torch.zeros(1, self.dim))
        self.register_buffer('batch_scale', torch.ones(1, self.dim))

        assert isinstance(affine, bool)
        self.register_buffer('_affine', torch.tensor(affine))

        # We'll save these two parameters even if _affine is not True
        # because, otherwise, when we load the flow,
        # if affine has not the same value as the state_dict, 
        # it will raise an Exception.
        self.loc = nn.Parameter(torch.zeros(1, self.dim))
        self.log_scale = nn.Parameter(torch.zeros(1, self.dim))

    def _warm_start(self, x, **kwargs):
        with torch.no_grad():
            self.batch_loc.data = x.mean(0, keepdim=True)
            self.batch_scale.data = x.std(0, keepdim=True) + self.eps

        super()._warm_start(x, **kwargs)

    def _batch_stats(self, x=None, update=None):
        if self.training and x is not None:
            assert x.size(0) >= 2, \
                'If training BatchNorm, pass more than 1 sample.'

            bloc = x.mean(0, keepdim=True)
            bscale = x.std(0, keepdim=True) + self.eps

            # Update self.batch_loc, self.batch_scale
            with torch.no_grad():
                if not self.initialized:
                    self.batch_loc.data = bloc
                    self.batch_scale.data = bscale
                    
                    self.initialized = True
                else:
                    m = self.momentum
                    self.batch_loc.data = (1 - m) * self.batch_loc + m * bloc
                    self.batch_scale.data = \
                        (1 - m) * self.batch_scale + m * bscale

        else:
            bloc, bscale = self.batch_loc, self.batch_scale

        loc, scale = self.loc, self.log_scale

        scale = torch.exp(scale) + self.eps
        # Note that batch_scale does not use activation,
        # since it is already in scale units.

        return bloc, bscale, loc, scale

    def _log_abs_det(self, bscale):
        if self.affine:
            return (self.log_scale - torch.log(bscale)).sum(dim=1)
        else:
            return -torch.log(bscale).sum(dim=1)

    def _transform(self, x, *h, log_abs_det=False, **kwargs):
        bloc, bscale, loc, scale = self._batch_stats(x)
        u = (x - bloc) / bscale 
        
        if self.affine:
            u = u * scale + loc
        
        if log_abs_det:
            log_abs_det = self._log_abs_det(bscale)
            return u, log_abs_det
        else:
            return u

    def _invert(self, u, *h, log_abs_det=False, **kwargs):
        bloc, bscale, loc, scale = self._batch_stats()
        
        if self.affine:
            x = (u - loc) / scale * bscale + bloc
        else:
            x = u * bscale + bloc
        
        if log_abs_det:
            log_abs_det = -self._log_abs_det(bscale)
            return x, log_abs_det
        else:
            return x


class Shuffle(Flow):
    """Perform a dimension-wise permutation."""
    
    def __init__(self, perm=None, **kwargs):
        """
        Args:
            perm (torch.Tensor): permutation to apply.
        """
        
        super().__init__(**kwargs)
        
        if perm is None:
            perm = torch.randperm(self.dim)
                
        assert perm.shape == (self.dim,)
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', torch.argsort(self.perm))
        
    def _log_abs_det(self, x):
        # By doing a permutation, det is always 1 or -1. 
        # Hence, log|det| is always 0.
        return torch.zeros_like(x[:, 0])
        
    def _transform(self, x, log_abs_det=False, **kwargs):
        u = x[:, self.perm]
        
        if log_abs_det:
            return u, self._log_abs_det(x)
        else:
            return u
        
    def _invert(self, u, log_abs_det=False, **kwargs):
        x = u[:, self.inv_perm]
        
        if log_abs_det:
            return x, -self._log_abs_det(x)
        else:
            return x