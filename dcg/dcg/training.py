"""
Train utilities for Deep Causal Graphs.

Includes functions:

* `get_device`: get the default torch.device (cuda if available).
* `train`: used to train graphs with early stopping.
* `plot_losses`: plot training and validation losses from a `train` session.
* `test_nll`: compute the negative-log-likelihood of the test set.
* `test_loglk`: compute the log-likelihood of the test set.
"""

from torch_misc.training import get_device, train, plot_losses, average_loss, data_loader
from torch.utils.data import TensorDataset

def loss_f(ex_n=100, target_node=None):
    """Create a training loss based on graph.nll.
    
    Assumes that batch is a tuple (tensor,).
    Sends tensor to graph.device and calls nll.
    
    Call this function to set the ex_n value; 
    returns a loss function with the set ex_n.
    """
    
    def f(graph, batch):
        tensor, = batch
        tensor = tensor.to(graph.device)
        
        return graph.nll(tensor, target_node=target_node, ex_n=ex_n)
    
    return f

def test_nll(graph, tensor, target_node=None, batch_size=1024, ex_n=100):
    loader = data_loader(TensorDataset(tensor), batch_size, num_samples=None, drop_last=False)
    
    return average_loss(graph, loader, loss_f(ex_n=ex_n, target_node=target_node))

def test_loglk(graph, tensor, target_node=None, batch_size=1024, ex_n=100):
    return -test_nll(graph, tensor, target_node=target_node, batch_size=batch_size, ex_n=ex_n)