"""
Provides miscellaneous utility functions.
"""

from torch_misc.modules import Module, requires_init, AdjacencyMaskedNet
from torch_misc.prob import log_sum_exp_trick, log_mean_exp_trick
from torch_misc.misc import topological_order
from torch_misc.utils import softplus_inv