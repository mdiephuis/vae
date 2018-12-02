from __future__ import print_function
import numpy as np
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchvision.utils import save_image

from vae_models import CVAE

from nn_helpers.losses import loss_bce_kld, EarlyStopping
from nn_helpers.utils import init_weights, one_hot
from nn_helpers.utils import type_tfloat
from nn_helpers.visdom_grapher import VisdomGrapher
from nn_helpers.data import Loader


parser = argparse.ArgumentParser(description='VAE example')

# Task parameters
parser.add_argument('--uid', type=str, default='VAE',
                    help='Staging identifier (default:VAE)')

# Model parameters

parser.add_argument('--conditional', action='store_true', default=True,
                    help='Enable CVAE')
parser.add_argument('--latent-size', type=int, default=20, metavar='N',
                    help='VAE latent size (default: 20')

parser.add_argument('--encoder-size', type=int, default=200, metavar='N',
                    help='VAE encoder size (default: 200')

# data loader parameters
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset (default: none')
parser.add_argument('--data-dir', type=str, default=None,
                    help='directory to load data from (default: none')
parser.add_argument('--download-data', action='store_true', default=False,
                    help='Automatically download dataset (default: false)')
parser.add_argument('--batch-size', type=int, default=23, metavar='N',
                    help='input training batch-size')


# Optimizer
parser.add_argument('--optimizer', type=str, default="adam",
                    help='Optimizer (default: Adam')
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

# Enable CUDA, set tensor type and device
if args.cuda:
    dtype = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
    print('GPU')
else:
    dtype = torch.FloatTensor
    device = torch.device("cpu")

# Handle randomization
if args.seed is not None:
    print('Seed: {}'.format(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


def reconstruction_example(model, data_loader):

    model.eval()
    num_class = data_loader.num_class
    img_shape = data_loader.img_shape[1:]

    for _, (x, y) in enumerate(data_loader.test_loader):
        y = one_hot(y, num_class)
        y = y.type(type_tfloat(args.cuda))
        x_hat, _, _ = model(x, y)
        break

    x = x[:num_class].cpu().view(num_class * img_shape[0], img_shape[1])
    x_hat = x_hat[:num_class].cpu().view(num_class * img_shape[0], img_shape[1])
    comparison = torch.cat((x, x_hat), 1).view(num_class * img_shape[0], 2 * img_shape[1])
    return comparison


def latentspace_example(model, latent_size, data_loader):
    num_class = data_loader.num_class
    img_shape = data_loader.img_shape[1:]

    draw = torch.randn(num_class, latent_size, )
    label = torch.eye(num_class, num_class, )
    sample = model.decode(draw, label).cpu().view(num_class, 1, img_shape[0], img_shape[1])
    return sample


# save checkpoint
def save_checkpoint(state, filename):
    torch.save(state, filename)


def get_optimizer(model):
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    optimizer = Adam(model.parameters(), lr=lr,
                     betas=(beta1, beta2))
    return optimizer


def execute_graph(model, conditional, data_loader, loss_fn, scheduler, optimizer, use_visdom):
    # Training loss
    t_loss = train_validate(model, data_loader, loss_bce_kld, optimizer, conditional, train=True)

    # Validation loss
    v_loss = train_validate(model, data_loader, loss_bce_kld, optimizer, conditional, train=False)

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
        sample = latentspace_example(model, latent_size, data_loader)
        sample = sample.detach().numpy()
        vis.add_image('Generated sample ' + str(epoch), 'generated', sample)

        # Visdom: Show example reconstruction from the test set
        comparison = reconstruction_example(model, data_loader)
        comparison = comparison.detach().numpy()
        vis.add_image('Reconstruction sample ' + str(epoch), 'recon', comparison)

    return v_loss


def train_validate(model, data_loader, loss_fn, optimizer, conditional, train):
    model.train() if train else model.eval()
    loader = data_loader.train_loader if train else data_loader.test_loader

    batch_loss = 0
    batch_size = data_loader.batch_size
    num_class = data_loader.num_class

    for batch_idx, (x, y) in enumerate(loader):
        loss = 0
        if train:
            opt.zero_grad()
        if conditional:
            y = one_hot(y, num_class)
            y = y.type(type_tfloat(args.cuda))

            x_hat, mu, log_var = model(x, y)
        else:
            x_hat, mu, log_var = model(x)

        loss = loss_fn(x, x_hat, mu, log_var)

        batch_loss += loss.item() / batch_size

        if train:
            loss.backward()
            optimizer.step()
    # collect better stats
    return batch_loss / (batch_idx + 1)


"""
Visdom init
"""
if use_visdom:
    vis = VisdomGrapher(args.uid, args.visdom_url, args.visdom_port)

"""
Get the dataloader
"""
data_loader = Loader(args.dataset_name, args.data_dir, args.download_data, True, args.batch_size, None, None, args.cuda)


"""
Run a conditional one-hot VAE, pass latent_dim, input_shape and num_class
"""
input_shape = data_loader.img_shape
num_class = data_loader.num_class
encoder_size = args.encoder_size
latent_size = args.latent_size

model = CVAE(input_shape, encoder_size, latent_size, num_class).type(dtype)
model.apply(init_weights)

opt = get_optimizer(model)
scheduler = ReduceLROnPlateau(opt, 'min', verbose=True)
early_stopping = EarlyStopping('min', 0.0005, 15)

num_epochs = args.epochs
conditional = True
best_loss = np.inf
# Main training and validation loop

for epoch in range(1, num_epochs + 1):
    v_loss = execute_graph(model, conditional, data_loader,
                           loss_bce_kld, scheduler, opt, use_visdom)

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
sample = latentspace_example(model, latent_size, data_loader)
save_image(sample, 'output/sample_' + str(num_epochs) + '.png')

# Make a final reconstruction, and write to disk
comparison = reconstruction_example(model, data_loader)
save_image(comparison, 'output/comparison_' + str(num_epochs) + '.png')
