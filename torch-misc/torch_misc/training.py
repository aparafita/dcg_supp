"""
Train utilities for flows.

Includes functions:

* `get_device`: get the default torch.device (cuda if available).
* `train`: used to train flows with early stopping.
* `plot_losses`: plot training and validation losses from a `train` session.
* `test_nll`: compute the test negative-loglikelihood of the test set.
"""


from tempfile import TemporaryFile
from collections import OrderedDict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def get_device():
    """Return default cuda device if available, cpu otherwise."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NanLoss(Exception):
    
    def __init__(self, epoch, training):
        self.epoch = epoch
        self.training = training
        
    def __str__(self):
        return f'NaN loss during {"training" if self.training else "validation"} at epoch {self.epoch}'

def train(
    module, train_loader, val_loader, loss_f,
    optimizer=optim.AdamW, optimizer_kwargs={},
    scheduler=None, scheduler_kwargs={},
    n_epochs=None, patience=None, 
    gradient_clipping=None,
    callback=None, use_tqdm=True,
):
    r"""Train a module for a certain number of epochs and/or with early stopping.
    
    If a training or validation loss returns any nan or inf, halts training.

    Can KeyboardInterrupt safely;
    the resulting model will be the best one before the interruption.

    Args:
        module (nn.Module): module to train.
        
        train_loader (torch.utils.data.DataLoader): DataLoader for training.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation.
        
        loss_f (function): function loss_f(module, batch),
            where module is the Module to be trained and
            batch is the result of the DataLoader.
            Must return a tensor with the loss for every sample in the batch.
        
        optimizer (torch.optim.Optimizer): optimizer class to use.
        optimizer_kwargs (dict): kwargs to pass to the optimizer.
        
        scheduler (torch.optim.lr_scheduler._LRScheduler): 
            LR scheduler to use. 
            Its step will be called at the end of every epoch.
        scheduler_kwargs (dict): kwargs to pass to the scheduler.

        n_epochs (int): maximum number of epochs for training.
            If None, runs forever until patience interrupts training.
        patience (int): maximum number of epochs with no improvement
            in validation loss before stopping. 
            If None, runs until n_epochs have passed.
            
        gradient_clipping (float or None): gradient clipping to apply during training.
            If None, no gradient clipping will be applied.
            
        callback (function or None): if not None, callback to call during every event.
            Its signature is: f(event_type::str, epoch::int),
            where event_type is one of the following:
                - train_start: start a training epoch.
                - train_step: completed a batch from training.
                - train_end: end a training epoch.
                - val_start: start of a validation epoch.
                - val_end: end of a validation epoch.
            
        use_tqdm (bool): whether to use tqdm or not to inform of training progress.
    
    Note that either n_epochs or patience must not be None. Both can be set though.

    Returns:
        train_losses: list with entries (epoch, mean(loss)).
        val_losses: list with entries (epoch, loss).

    The results of this function can be passed to `plot_losses` directly.
    """
        
    assert n_epochs is None or n_epochs > 0
    assert patience is None or patience > 0
    assert n_epochs is not None or patience is not None
    assert gradient_clipping is None or gradient_clipping > 0
    
    optimizer = optimizer(module.parameters(), **optimizer_kwargs)
    
    if scheduler is not None:            
        scheduler = scheduler(optimizer, **scheduler_kwargs)

    train_losses, val_losses = [], []
    
    val_loss = np.inf
    best_loss = np.inf
    best_epoch = 0
    best_model = TemporaryFile()
    
    # Define an iterator for epochs; if n_epochs is None, never finishes
    def epoch_it():
        if n_epochs is not None:
            yield from range(1, n_epochs + 1)
        else:
            epoch = 1
            while True:
                yield epoch
                epoch += 1

    if use_tqdm:
        # Open tqdm as tq, it will be closed at the finally block
        tq = tqdm(total=n_epochs)
        
    start_time = datetime.now()
        
    try:
        for epoch in epoch_it():
            # Train
            module.train()
            
            if callback is not None: callback('train_start', epoch)

            current_losses = []
            for n, batch in enumerate(train_loader):
                loss = loss_f(module, batch)

                size = len(loss)
                loss = loss.mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    raise NanLoss(epoch, module.training)

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if gradient_clipping is not None:
                    nn.utils.clip_grad_norm_(
                        module.parameters(), 
                        gradient_clipping
                    )

                optimizer.step()

                current_losses.append((loss.item(), size))

                if use_tqdm:
                    tq.set_postfix(
                        OrderedDict(
                            prog='%.3d%%' % (n / len(train_loader) * 100), 
                            best='[%d: %.3e]' % (best_epoch, best_loss)
                        )
                    )
                    
                if callback is not None: callback('train_step', epoch)

            total_size = sum(size for _, size in current_losses)
            loss = sum(
                loss * (size / total_size)
                for loss, size in current_losses
            )
            train_losses.append((epoch, loss))

            if scheduler is not None:
                scheduler.step()

            if callback is not None: callback('train_end', epoch)

            # Validation
            module.eval()
            
            if callback is not None: callback('val_start', epoch)
            
            val_loss = average_loss(module, val_loader, loss_f)
            val_losses.append((epoch, val_loss))

            if np.isnan(val_loss) or np.isinf(val_loss):
                raise NanLoss(epoch, module.training)

            # Early stopping
            if best_loss > val_loss:
                best_loss = val_loss
                best_epoch = epoch

                best_model.seek(0)
                torch.save(module.state_dict(), best_model)

            if use_tqdm:
                tq.update()
                tq.set_postfix(
                    OrderedDict(
                        prog='100%', 
                        best='[%d: %.3e]' % (best_epoch, best_loss)
                    )
                )
                
            if callback is not None: callback('val_end', epoch)

            if patience is not None and epoch - best_epoch >= patience:
                break
                    
    except NanLoss as e:
        print(e)
        pass # halt training without losing everything
    except KeyboardInterrupt:
        print('Interrupted at epoch', epoch)
        pass # halt training without losing everything
    finally:
        if use_tqdm:
            tq.close()
            
        end_time = datetime.now()
        print('Training time: %d seconds' % (end_time - start_time).seconds)

    # Load best model before exiting
    # Note that it only exist if we're past the first epoch
    assert epoch > 1, 'Couldn\'t finish first epoch'
    best_model.seek(0)
    module.load_state_dict(torch.load(best_model))
    best_model.close()

    module.eval() # pass to eval mode before returning

    return train_losses, val_losses


def plot_losses(*losses, cellsize=(6, 4), titles=None, max_nc=4):
    """Plot train and validation losses from a `train` call.

    Args:
        train_losses (list): (epoch, loss) pairs to plot for training.
        val_losses (list): (epoch, loss) pairs to plot for validation.
        cellsize (tuple): (width, height) for each cell in the plot.
    """
    
    N = len(losses)
    assert N
    if titles is None:
        if N == 2:
            titles = ['train_loss', 'val_loss']
        else:
            titles = [None] * N
    else:
        assert len(titles) == len(losses)
            
    nc = min(N, max_nc)
    nr = (N - 1) // nc + 1
    
    w, h = cellsize
    fig, axes = plt.subplots(nr, nc, figsize=(w * nc, h * nr), squeeze=False)
    axes = axes.flatten()
    
    for ax in axes[N:]:
        ax.axis('off')
        
    for ax, l, title in zip(axes, losses, titles):
        best_epoch, best_loss = min(l, key=lambda pair: pair[1])
        
        ax.plot(*np.array(l).T)
        ax.axvline(best_epoch, ls='dashed', color='gray')
        
        if title is not None:
            ax.set_title('%s (best %.3e)' % (title, best_loss))
            

'''
def average_loss(module, data_loader, loss_f):
    """Compute the average loss for a complete pass over the given DataLoader.
    
    Note that this won't compute any gradients,
    as it will run in torch.no_grad() mode.
    
    Returns a float (not a Tensor).

    Args:
        module (nn.Module): module to train.
        data_loader (torch.utils.data.DataLoader): DataLoader to use.
    """
                      
    # Note that since we don't know how many samples there will be,
    # and the loss sum could end up being too big, 
    # we'll divide at each step by the abs max
    # to keep the estimate stable.
    
    old_M = 0

    with torch.no_grad(): # won't accumulate info about gradient
        loss_sum, size = 0., 0
        for batch in data_loader:
            loss = loss_f(module, batch)
            
            new_M = max(old_M, loss.abs().max().item()) + 1e-6
            loss_sum = loss_sum * (old_M / new_M) + (loss / new_M).sum().item()
            
            size += len(loss)
            old_M = new_M
            
            del loss

    return loss_sum * (old_M / size)
'''

def average_loss(module, data_loader, loss_f, std=False):
    """Compute the average loss for a complete pass over the given DataLoader.
    
    Note that this won't compute any gradients,
    as it will run in torch.no_grad() mode.
    
    Returns a float (not a Tensor). 
    If std is True, also returns the standard deviation of the losses.

    Args:
        module (nn.Module): module to train.
        data_loader (torch.utils.data.DataLoader): DataLoader to use.
    """
                      
    # Note that since we don't know how many samples there will be,
    # and the loss sum could end up being too big, 
    # we'll divide at each step by the abs max
    # to keep the estimate stable.
    
    losses = []

    with torch.no_grad(): # won't accumulate info about gradient
        for batch in data_loader:
            loss = loss_f(module, batch)
            
            losses.append(loss.cpu().numpy())
            del loss
            
    losses = np.concatenate(losses, 0)
    
    if not std:
        return losses.mean()
    else:
        return losses.mean(), losses.std() / np.sqrt(len(losses))

def data_loader(dataset, batch_size, num_samples=None, drop_last=True):
    if num_samples is None:
        sampler = None
        shuffle = True
    else:
        if isinstance(num_samples, float):
            num_samples = int(num_samples * len(dataset))
            
        assert 0 < num_samples and num_samples <= len(dataset)
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        shuffle = False
            
    return DataLoader(
        dataset, sampler=sampler, shuffle=shuffle,
        batch_size=batch_size, drop_last=drop_last
    )

if __name__ == '__main__':
    import sys
    args = sys.argv
    
    if len(args) >= 2 and args[1] == '-h':
        print('python -m flow.training plot [train_losses.npy] [val_losses.npy]')
    
    if len(args) == 4 and args[1] == 'plot':
        train, val = args[2:4]
        train_losses, val_losses = np.load(train), np.load(val)
        
        plot_losses(train_losses, val_losses)
        plt.show()