"""
Provides sampling functions, mainly used by LatentNodes.

* `uniform`
* `gumbel`
* `categorical`
* `bernoulli`
* `rejection_sampling`
"""

import warnings
import torch


def uniform(*shape, device=torch.device('cpu'), eps=1e-6):
    """Sample the given shape from the Uniform(eps, 1 - eps) distribution.

    Args:
        *shape (int): shape of the tensor.
        device (torch.device): where to project the tensor.
        eps (float): minimum value of the uniform. Used to avoid 0s and 1s.
    """
    assert shape
    assert 0 < eps and eps < .5

    x = torch.rand(*shape, device=device)
    x = x * (1 - eps * 2) + eps

    return x


def gumbel(*shape, device=torch.device('cpu'), eps=1e-6):
    """Sample from the Gumbel(0, 1) distribution."""

    assert shape
    assert 0 < eps and eps < .5

    return -torch.log(-torch.log(uniform(*shape, device=device, eps=eps)))


def categorical(n, log_p, gumbel_sample=None, soft=False, temperature=1e-2):
    r"""Sample from a Categorical distribution using the Gumbel trick.

    Using a Gumbel(0, 1) sample, samples from the specified 
    Categorical(log_p) distribution. 

    Its result can be soft ($x \in (0, 1)$) or not ($x \in {0, 1}$).
    
    Args:
        n (int): how many samples to sample.
        log_p (torch.Tensor): tensor with the log-probabilities of each class.
        gumbel_sample (torch.Tensor): if not None, uses the given tensor
            as the sample from the Gumbel distribution. Otherwise, 
            samples a Gumbel sample automatically.
        soft (bool): whether to return soft samples or hard samples.
            Even when they are hard, their gradient is the same as the soft one.
        temperature (float): temperature hyperparameter 
            to use for soft samples in the softmax step. 
            Lower values (closer to 0) lead to "harder" values.
    """

    if len(log_p.shape) == 1:
        log_p = log_p.unsqueeze(0)

    assert not n % log_p.size(0)

    if gumbel_sample is None:
        gumbel_sample = gumbel(n, log_p.size(1), device=log_p.device)

    assert gumbel_sample.shape == (n, log_p.size(1))

    y = (log_p + gumbel_sample)

    if soft:
        y = torch.softmax(y / temperature, dim=1)
    else:
        y_hard = torch.scatter(
            torch.zeros_like(y), 1, 
            y.argmax(1, keepdim=True), 1
        )

        y = (y_hard - y).detach() + y # this preserves the gradient while being hard

    return y


def bernoulli(n, p, gumbel_sample=None, soft=False, temperature=1e-2):
    """Use `categorical` to sample Bernoulli samples.

    Returns:
        sample: tensor of shape (n, 1) with the Bernoulli samples.
    """
    if len(p.shape) == 1:
        p = p.unsqueeze(1)

    assert not n % p.size(0) and p.size(1) == 1
    p = p.repeat(n // p.size(0), 1) # broadcast

    log_p = torch.log(torch.cat([1 - p, p], 1))
    
    return categorical(
        n, log_p, 
        gumbel_sample=gumbel_sample, 
        soft=soft, 
        temperature=temperature
    )[:, [1]]


def weighted_sampling(N, x, log_w):
    """Sample N subsamples from x with replacement using log-weights log_w."""
    idx = categorical(N, log_w, soft=False) > .5
    return x.unsqueeze(0).repeat(N, *([1] * len(x.shape)))[idx.unsqueeze(2)]


class RejectionSamplingException(Exception):
    """Exception raised when `rejection_sampling`
    can't find a sample before n steps.

    Has two attributes, z and idx, both tensors,
    that contain the rejection_sampling samples obtained up to that point and 
    a boolean index showing which rows were still rejected, respectively.
    """

    def __init__(self, z, idx):
        """Constructor for RejectionSamplingException.

        Args:
            z (torch.Tensor): samples before raising the Exception.
            idx (torch.Tensor): indexes of those samples 
                that were rejected before raising the Exception.
        """
        self.z = z
        self.idx = idx


def rejection_sampling(
    n, dim, *args, 
    sampling=None, rejection=None, max_rejections=0, 
    warn=100, device=torch.device('cpu'), **kwargs
):
    """Sample using rejection sampling.

    Samples all required samples and, 
    if any does match the rejection property,
    they are resampled again.

    Args:
        n (int): number of samples.
        dim (int): dimensionality of each sample.
        *args (anything): positional arguments passed 
            to the sampling and rejection functions.
        sampling (function): function(n, *args, device=None, **kwargs)
            that samples n samples to the given device, using, 
            optionally, the given *args and **kwargs.
        rejection (function): function(z, *args, **kwargs)
            that returns a boolean tensor showing which samples in z to reject,
            using, optionally, the given *args and **kwargs.
        max_rejections (int): whenever a sample has been rejected more
            than max_rejections times, the function will stop and
            raise a `RejectionSamplingException`. If 0, no maximum is applied.
        warn (int): when a sample has been rejected more than warn times,
            the function will issue a warning. If 0, no warning is raised.
        device (torch.device): where to map the newly generated samples.
        **kwargs (anything): keyword arguments passed
            to the sampling and rejection functions.

    Note that if any argument in *args or **kwargs is a Tensor, 
    it will be broadcasted and filtered to match the number of samples
    still remaining to be sampled.

    Returns:
        z (torch.Tensor)
            tensor with the resulting samples.

    """

    assert sampling is not None and rejection is not None
    assert max_rejections >= 0
           
    z = torch.zeros(n, dim, device=device)
    to_fill = torch.ones(n, dtype=torch.bool, device=device)

    with torch.no_grad():
        rejections = 0

        while to_fill.any():
            # First, check that we haven't rejected too many times
            if warn and rejections == warn:
                warnings.warn(f'Exceeded max rejections: {warn}')
            if max_rejections and rejections == max_rejections:
                raise RejectionSamplingException(z, to_fill)

            # Then, start sampling            
            # Filter *args and **kwargs by to_fill
            f = lambda x, to_fill: (
                x.repeat(len(to_fill), 1) 
                if x.shape[0] == 1 # broadcast
                else x
            )[to_fill]

            f_args = tuple(
                f(arg, to_fill) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            )
            f_kwargs = {
                k: f(v, to_fill) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }

            # Compute the new values
            n = to_fill.sum()
            new_z = sampling(n, *f_args, device=device, **f_kwargs)
            assert new_z.shape == (n, dim), (new_z.shape, (n, dim))
            z[to_fill] = new_z

            # Check if they should be rejected
            to_fill[to_fill] = rejection(new_z, *f_args, **f_kwargs)

            rejections += 1

    return z