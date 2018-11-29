"""
    Python (C)VAE implementations
    Maurits Diephuis
"""

from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from vae_models import CVAE

from nn_helpers.losses import loss_bce_kld, EarlyStopping
from nn_helpers.utils import one_hot_np, init_weights
from nn_helpers.visdom_grapher import VisdomGrapher


parser = argparse.ArgumentParser(description='VAE example')

# Task parameters
parser.add_argument('--uid', type=str, default='VAE',
                    help='Staging identifier (default:VAE)')

# Model parameters

parser.add_argument('--conditional', action='store_true', default=True,
                    help='Enable CVAE')

# Optimizer
parser.add_argument('--optimizer', type=str, default="adam",
                    help='Optimizer (default: Adam')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of training epochs')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4')

# Visdom / tensorboard
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom url, needs http, e.g. http://localhost (default: None)')
parser.add_argument('--visdom-port', type=int, default=8097,
                    help='visdom server port (default: 8097')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batch interval for logging (default: 1')

# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

parser.add_argument('--ngpu', type=int, default=1,
                    help='Number of gpus available (default: 1)')

parser.add_argument('--seed', type=int, default=None,
                    help='Seed for numpy and pytorch (default: None')


args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
use_visdom = args.visdom_url is not None


# Handle randomization


# Enable CUDA, set tensor type and device
# todo: refractor this

if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")


if args.seed is not None:
    print('Seed: {}'.format(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


def reconstruction_example(model, device, dtype):
    model.eval()
    for _, (x, y) in enumerate(loader_val):
        x = x.type(dtype)
        x = x.to(device)
        y = torch.from_numpy(one_hot_np(y, 10))
        y = y.type(dtype)
        y = y.to(device)
        x_hat, _, _ = model(x, y)
        break

    x = x[:10].cpu().view(10 * 28, 28)
    x_hat = x_hat[:10].cpu().view(10 * 28, 28)
    comparison = torch.cat((x, x_hat), 1).view(10 * 28, 2 * 28)
    return comparison


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
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

loader_val = DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


def get_optimizer(model):
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    optimizer = Adam(model.parameters(), lr=lr,
                     betas=(beta1, beta2))
    return optimizer


def execute_graph(model, conditional, loader_train, loader_val,
                  loss_fn, scheduler, use_visdom):
    # Training loss
    t_loss = train_validate(model, loader_train, loss_bce_kld, scheduler, conditional, train=True)

    # Validation loss
    v_loss = train_validate(model, loader_val, loss_bce_kld, scheduler, conditional, train=False)

    # Step the scheduler based on the validation loss
    scheduler.step(v_loss)

    print('====> Epoch: {} Average Train loss: {:.4f}'.format(
          epoch, t_loss))
    print('====> Epoch: {} Average Validation loss: {:.4f}'.format(
          epoch, v_loss))

    if use_visdom:
        # Visdom: update training and validation loss plots
        vis.add_scalar('Training loss', idtag='train', y=t_loss, x=epoch)
        vis.add_scalar('Validation loss', idtag='valid', y=v_loss, x=epoch)

        # Visdom: Show generated images
        sample = latentspace_example(model, device)
        sample = sample.detach().numpy()
        vis.add_image('Generated sample ' + str(epoch), 'generated', sample)

        # Visdom: Show example reconstruction
        comparison = reconstruction_example(model, device, dtype)
        comparison = comparison.detach().numpy()
        vis.add_image('Reconstruction sample ' + str(epoch), 'recon', comparison)

    return v_loss


def train_validate(model, loader_data, loss_fn, scheduler, conditional, train):
    model.train() if train else model.eval()
    batch_loss = 0
    batch_sz = len(loader_data.dataset)
    for batch_idx, (x, y) in enumerate(loader_data):
        x = x.to(device)
        if train:
            opt.zero_grad()
        if conditional:
            # use new function
            y = torch.from_numpy(one_hot_np(y, 10))
            # refractor out
            y = y.type(dtype)
            y = y.to(device)
            x_hat, mu, log_var = model(x, y)
        else:
            x_hat, mu, log_var = model(x)

        loss = loss_fn(x, x_hat, mu, log_var)

        batch_loss += loss.item()

        if train:
            loss.backward()
            opt.step()
    # collect better stats
    return batch_loss / batch_sz


"""
Visdom init
"""
if use_visdom:
    vis = VisdomGrapher(args.uid, args.visdom_url, args.visdom_port)

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
    v_loss = execute_graph(model, conditional, loader_train, loader_val,
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
