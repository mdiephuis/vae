"""
    Python (C)VAE implementation for CIFAR 10
    Maurits Diephuis
"""

from __future__ import print_function
import numpy as np
import time
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import init
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from visdom import Visdom

from vae_models import CVAE_C10
from net_utils import EarlyStopping

parser = argparse.ArgumentParser(description='VAE example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of training epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable cuda')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batch interval for logging')
parser.add_argument('--conditional', action='store_true', default=True,
                    help='Enable CVAE')
parser.add_argument('--no-visdom', action='store_true', default=False,
                    help='Enables visdom output.')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
use_visdom = not args.no_visdom

# Enable CUDA, set tensor type and device
if args.cuda :
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


# Visdom hooks
def update_plot(win_title, y, x, opts={}):
    '''
    Update vidomplot by win_title. If it doesn't exist, create a new plot
    - win_title: name and title of plot
    - y: y coord
    - x: x coord
    - options_dict, example {'legend': 'NAME', 'ytickmin': 0, 'ytickmax': 1}
    '''
    if not viz.win_exists(win_title):
        viz.line(Y=np.array([y]), X=np.array([x]), win=win_title,
                 opts=opts)
    else:
        viz.line(Y=np.array([y]), X=np.array([x]), win=win_title,
                 update='append', opts=opts)


# Fix for dims, cifar
def reconstruction_example(model, device, dtype):
    model.eval()
    for _, (x, y) in enumerate(loader_val):
        x = x.type(dtype)
        x = x.to(device)

        y = torch.from_numpy(one_hot(y))
        y = y.type(dtype)
        y = y.to(device)
        x_hat, _, _ = model(x, y)
        break

    x = x[:10].cpu().view(10*28, 28)
    x_hat = x_hat[:10].cpu().view(10*28, 28)
    comparison = torch.cat((x, x_hat), 1).view(10*28, 2*28)
    return comparison


# Fix for dims, cifar
def latentspace_example(model, device):
    draw = torch.randn(10, 20, device=device)
    label = torch.eye(10, 10, device=device)
    sample = model.decode(draw, label).cpu().view(10, 1, 28, 28)
    return sample


# save checkpoint
def save_checkpoint(state, filename):
    torch.save(state, filename)


# Training and Validation Dataloaders
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

loader_train = DataLoader(
    datasets.CIFAR10('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

loader_val = DataLoader(
    datasets.CIFAR10('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)


def one_hot(labels, n_class=10):
    vec = np.zeros((labels.shape[0], n_class))
    for ind, label in enumerate(labels):
        vec[ind, label] = 1
    return vec


def get_optimizer(model):
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    optimizer = Adam(model.parameters(), lr=lr,
                                 betas=(beta1, beta2))
    return optimizer


def loss_bce_kld(x, x_hat, mu, log_var):
    """
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    BCE = F.binary_cross_entropy(x_hat, x.view(-1, 784), size_average=True)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return KLD + BCE


def train_validate(model, conditional, loader_train, loader_val,
                   loss_fn, scheduler, use_visdom):
    # Training loss
    t_loss = train(model, conditional, loader_train, loss_bce_kld, scheduler)

    # Validation loss
    v_loss = validate(model, conditional, loader_val, loss_bce_kld, scheduler)

    # Step the scheduler
    scheduler.step(v_loss)

    print('====> Epoch: {} Average Train loss: {:.4f}'.format(
          epoch, t_loss))
    print('====> Epoch: {} Average Validation loss: {:.4f}'.format(
          epoch, v_loss))
    
    if use_visdom:
        # Visdom: update training and validation loss plots
        update_plot('tloss',  y=t_loss, x=epoch,
                    opts=dict(title='Training loss'))

        update_plot('vloss', y=v_loss, x=epoch,
                    opts=dict(title='Validation loss'))

        # Visdom: Show generated images
        sample = latentspace_example(model, device)
        sample = sample.detach().numpy()
        viz.images(sample, win='gen',
                opts=dict(title='Generated sample ' + str(epoch)))

        # Visdom: Show example reconstruction
        comparison = reconstruction_example(model, device, dtype)
        viz.images(comparison.detach().numpy(), win='recon',
                opts=dict(title='Reconstruction ' + str(epoch)))

    return v_loss


def train(model, conditional, loader_data, loss_fn, scheduler):
    model.train()
    train_loss = 0
    n_train = len(loader_data.dataset)
    for batch_idx, (x, y) in enumerate(loader_data):
        x = x.to(device)
        opt.zero_grad()
        if conditional:
            y = torch.from_numpy(one_hot(y))
            y = y.type(dtype)
            y = y.to(device)
            x_hat, mu, log_var = model(x, y)
        else:
            x_hat, mu, log_var = model(x)
        loss = loss_fn(x, x_hat, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        opt.step()
    return train_loss / n_train


def validate(model, conditional, loader_data, loss_fn, scheduler):
    model.eval()
    val_loss = 0
    n_val = len(loader_data.dataset)
    for batch_idx, (x, y) in enumerate(loader_data):
        x = x.to(device)
        if conditional:
            y = torch.from_numpy(one_hot(y))
            y = y.type(dtype)
            y = y.to(device)
            x_hat, mu, log_var = model(x, y)
        else:
            x_hat, mu, log_var = model(x)
        val_loss += loss_fn(x, x_hat, mu, log_var).item()

    return val_loss / n_val


"""
Visdom init
"""
if use_visdom:
    env_name = 'VAE'
    viz = Visdom(env=env_name)
    startup_sec = 2
    while not viz.check_connection() and startup_sec > 0:
        time.sleep(0.1)
        startup_sec -= 0.1
    # assert viz.check_connection(), 'Visdom connection failed'

"""
Run a conditional one-hot VAE
"""
model = CVAE().type(dtype)
model.apply(init_weights)
opt = get_optimizer(model)
scheduler = ReduceLROnPlateau(opt, 'min', verbose=True)
early_stopping = EarlyStopping('min', 0.0005, 5)

num_epochs = args.epochs
conditional = True
best_loss = np.inf
# Main training and validation loop
for epoch in range(1, num_epochs + 1):
    v_loss = train_validate(model, conditional, loader_train, loader_val,
                            loss_bce_kld, scheduler, use_visdom)

    stop = early_stopping.step(v_loss)

    if v_loss < best_loss or stop:
        best_loss = v_loss
        print('Writing model checkpoint')
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'val_loss': v_loss
                        },
                        'models/CVAE_{:04.4f}.pt'.format(v_loss))
    if stop:
        print('Early stopping at epoch: {}'.format(epoch))
        break

# Write a final sample to disk
sample = latentspace_example(model, device)
save_image(sample, 'output/sample_' + str(num_epochs) + '.png')

# Make a final reconstrunction, and write to disk
comparison = reconstruction_example(model, device, dtype)
save_image(comparison, 'output/comparison_' + str(num_epochs) + '.png')
