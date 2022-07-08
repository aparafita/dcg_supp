import torch
import torch.nn.functional as F


def isin(x, values):
    """Return a boolean tensor with whether the values in x appear in the given set."""
    if not isinstance(values, torch.Tensor):
        values = torch.Tensor(list(values))
        
    values = values.to(x.device)
    return (x[..., None] == values).any(-1)


softplus = lambda x, eps=1e-6, **kwargs: F.softplus(x, **kwargs) + eps

def softplus_inv(x, eps=1e-6, threshold=20.):
    """Compute the softplus inverse."""
    x = x.clamp(0.)
    y = torch.zeros_like(x)

    idx = x < threshold
    # We deliberately ignore eps to avoid -inf
    y[idx] = torch.log(torch.exp(x[idx] + eps) - 1)
    y[~idx] = x[~idx]

    return y

logsigmoid = lambda x, alpha=1., **kwargs: -softplus(-alpha * x, **kwargs)