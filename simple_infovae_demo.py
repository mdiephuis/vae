from __future__ import print_function
import numpy as np
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchvision.utils import save_image

import torchvision.utils as tvu
from tensorboardX import SummaryWriter

from vae_models import INFO_VAE2
from vae_utils import reconstruction_example, generation_example, latentspace2d_example, save_checkpoint

from nn_helpers.losses import loss_bce_kld, EarlyStopping, loss_infovae
from nn_helpers.utils import init_weights, one_hot, to_cuda, type_tfloat, randn, eye
from nn_helpers.visdom_grapher import VisdomGrapher
from nn_helpers.data import Loader


parser = argparse.ArgumentParser(description='VAE example')

# Task parameters
parser.add_argument('--uid', type=str, default='VAE',
                    help='Staging identifier (default:VAE)')

# Model parameters
parser.add_argument('--latent-size', type=int, default=10, metavar='N',
                    help='VAE latent size (default: 10')

parser.add_argument('--out-channels', type=int, default=64, metavar='N',
                    help='VAE 2D conv channel output (default: 64')

parser.add_argument('--encoder-size', type=int, default=1024, metavar='N',
                    help='VAE encoder size (default: 1024')

# InfoVAE specific
parser.add_argument('--alpha', type=float, default=50.0, metavar='N',
                    help='InfoVAE alpha (default: 50.0')
parser.add_argument('--beta', type=float, default=51.0, metavar='N',
                    help='InfoVAE beta (default: 51.0')


# data loader parameters
parser.add_argument('--dataset-name', type=str, default='mnist',
                    help='Name of dataset (default: none')
parser.add_argument('--data-dir', type=str, default='data/',
                    help='directory to load data from (default: none')
parser.add_argument('--download-data', action='store_true', default=True,
                    help='Automatically download dataset (default: false)')
parser.add_argument('--batch-size', type=int, default=23, metavar='N',
                    help='input training batch-size')

# Optimizer
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of training epochs')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate (default: 1e-4')

# Visdom
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom url, needs http, e.g. http://localhost (default: None)')
parser.add_argument('--visdom-port', type=int, default=8097,
                    help='visdom server port (default: 8097')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batch interval for logging (default: 1')

# Log directory
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: logs)')


# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

parser.add_argument('--ngpu', type=int, default=1,
                    help='Number of gpus available (default: 1)')

parser.add_argument('--seed', type=int, default=None,
                    help='Seed for numpy and pytorch (default: None')


args = parser.parse_args()

# Set cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set visdom
use_visdom = args.visdom_url is not None

# Set tensorboard
use_tb = args.log_dir is not None
log_dir = args.log_dir

# Logger
if use_tb:
    logger = SummaryWriter()

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


def get_optimizer(model):
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    optimizer = Adam(model.parameters(), lr=lr,
                     betas=(beta1, beta2))
    return optimizer


def execute_graph(model, conditional, data_loader, loss_fn, scheduler, optimizer, use_visdom, use_tb):
    # Training los loss_fn
    t_loss = train_validate(model, data_loader, loss_fn,
                            optimizer, conditional, train=True)

    # Validation loss
    v_loss = train_validate(model, data_loader, loss_fn,
                            optimizer, conditional, train=False)

    # Step the scheduler based on the validation loss
    scheduler.step(v_loss)

    print('====> Epoch: {} Average Train loss: {:.4f}'.format(epoch, t_loss))
    print('====> Epoch: {} Average Validation loss: {:.4f}'.format(epoch, v_loss))

    if use_tb:
        # Training and validation loss
        logger.add_scalar(log_dir + '/validation-loss', v_loss, epoch)
        logger.add_scalar(log_dir + '/training-loss', t_loss, epoch)

        # todo: log gradient values of the model

        # image generation examples
        sample = generation_example(model, latent_size, data_loader, conditional, args.cuda)
        sample = sample.detach()
        sample = tvu.make_grid(sample, normalize=False, scale_each=True)
        logger.add_image('generation example', sample, epoch)

        # image reconstruction examples
        comparison = reconstruction_example(model, data_loader, conditional, args.cuda)
        comparison = comparison.detach()
        comparison = tvu.make_grid(comparison, normalize=False, scale_each=True)
        logger.add_image('reconstruction example', comparison, epoch)

                # latent space scatter example
        if args.latent_size == 2:
            z, labels = latentspace2d_example(model, data_loader, args.use_cuda)
            


    if use_visdom:
        # Visdom: update training and validation loss plots
        vis.add_scalar(t_loss, epoch, 'Training loss', idtag='train')
        vis.add_scalar(v_loss, epoch, 'Validation loss', idtag='valid')

        # Visdom: Show generated images
        sample = generation_example(model, latent_size, data_loader, conditional, args.cuda)
        sample = sample.detach().numpy()
        vis.add_image(sample, 'Generated sample ' + str(epoch), 'generated')

        # Visdom: Show example reconstruction from the test set
        comparison = reconstruction_example(model, data_loader, conditional, args.cuda)
        comparison = comparison.detach().numpy()
        vis.add_image(comparison, 'Reconstruction sample ' + str(epoch), 'recon')

    return v_loss


def train_validate(model, data_loader, loss_fn, optimizer, conditional, train):
    model.train() if train else model.eval()
    loader = data_loader.train_loader if train else data_loader.test_loader

    batch_loss = 0
    batch_size = data_loader.batch_size
    num_class = data_loader.num_class

    for batch_idx, (x, y) in enumerate(loader):
        loss = 0
        x = to_cuda(x) if args.cuda else x
        if train:
            opt.zero_grad()

        x_hat, mu_z, std_z = model(x)

        print(mu_z.size())

        loss = loss_fn(x, x_hat, mu_z, std_z, args.alpha, args.beta, args.cuda)

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
data_loader = Loader(args.dataset_name, args.data_dir,
                     args.download_data, True, args.batch_size, None, None, args.cuda)


"""
Model parameters
"""
input_shape = data_loader.img_shape
num_class = data_loader.num_class
encoder_size = args.encoder_size
decoder_size = args.encoder_size
latent_size = args.latent_size
out_channels = args.out_channels
conditional = False

# Model
model = INFO_VAE2(input_shape, out_channels,
                  encoder_size, latent_size).type(dtype)
model.apply(init_weights)

opt = get_optimizer(model)
scheduler = ReduceLROnPlateau(opt, 'min', verbose=True)
early_stopping = EarlyStopping('min', 0.0005, 15)

# Set loss function
# loss_fn = loss_bce_kld
loss_fn = loss_infovae

num_epochs = args.epochs

best_loss = np.inf
# Main training and validation loop

for epoch in range(1, num_epochs + 1):
    v_loss = execute_graph(model, conditional, data_loader,
                           loss_fn, scheduler, opt, use_visdom, use_tb)

    stop = early_stopping.step(v_loss)

    if v_loss < best_loss or stop:
        best_loss = v_loss
        print('Writing model checkpoint')
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'val_loss': v_loss
                        },
                        'models/INFOVAE_{:04.4f}.pt'.format(v_loss))
    if stop:
        print('Early stopping at epoch: {}'.format(epoch))
        break

# Write a final sample to disk
sample = generation_example(model, latent_size, data_loader, conditional, args.cuda)
save_image(sample, 'output/sample_' + str(num_epochs) + '.png')

# Make a final reconstruction, and write to disk
comparison = reconstruction_example(model, data_loader, conditional, args.cuda)
save_image(comparison, 'output/comparison_' + str(num_epochs) + '.png')

# 

# TensorboardX logger
logger.close()
