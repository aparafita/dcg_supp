"""
Probability related utilities.

Includes:

* `log_sum_exp_trick`: Computes `log(sum(w * exp(x), dim=dim, keepdim=keepdim))` safely.
* `log_mean_exp_trick`: Computes `log(mean(exp(x), dim=dim, keepdim=keepdim))` safely.
"""

import numpy as np
import torch


def log_sum_exp_trick(x, log_w=None, dim=-1, keepdim=False):
    r"""Computes `log(sum(w * exp(x), dim=dim, keepdim=keepdim))` safely.
    
    Uses the logsumexp trick for the computation of this quantity.

    Args:
        x (torch.Tensor): input tensor.
        log_w (torch.Tensor): logarithm of weights to apply. 
            If None, defaults to 0 (all weights 1).
            Must have same shape as x.
        dim (int): which dimension to aggregate.
        keepdim (bool): whether to preserve the aggregated dimension.
    """

    if log_w is None:
        log_w = torch.zeros_like(x)

    x = log_w + x # add log_w here so it affects M
    M = x.max(dim=dim, keepdim=True).values

    x = torch.log(torch.exp(x - M).sum(dim=dim, keepdim=True)) + M
    
    if not keepdim:
        x = x.squeeze(dim=dim)

    return x


def log_mean_exp_trick(x, dim=-1, keepdim=False):
    """Computes `log(mean(exp(x), dim=dim, keepdim=keepdim))` safely.

    Uses the logsumexp trick for the computation of this quantity.

    Args:
        x (torch.Tensor): input tensor.
        dim (int): which dimension to aggregate.
        keepdim (bool): whether to preserve the aggregated dimension.
    """
    N = x.size(dim)

    return log_sum_exp_trick(x, dim=dim, keepdim=keepdim) - np.log(N)