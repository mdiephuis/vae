import numpy as np
import torch
from nn_helpers.utils import one_hot, to_cuda, type_tfloat, randn, eye, zero_check_and_break
from nn_helpers.losses import loss_bce, kl_div_gaussian


def eval_data_nll(model, data_loader, sample_size, conditional, use_cuda):
    '''
    Evaluate the data negative log-likelihood for generative models
    '''
    model.eval()

    batch_size = data_loader.batch_size
    num_class = data_loader.num_class
    data_nll = []

    for x, y in data_loader.test_loader:
        x = to_cuda(x) if use_cuda else x
        indices = np.random.randint(0, x.size(0), size=sample_size)
        loss = []
        for ind in indices:
            x_sample = x[ind].unsqueeze(0)
            # fix this!
            # y = y[ind]
            # y = one_hot(y, num_class)
            # # expand to original batch size for comparison
            x_sample = x_sample.expand(batch_size, *x_sample.size()[1:]).contiguous()
            x_sample = x_sample.view(batch_size, -1)
            if conditional:
                x_hat, z_mu, z_std = model(x_sample, y)
            else:
                x_hat, z_mu, z_std = model(x)

            recon_loss = loss_bce(x, x_hat)
            # print("Recon loss {}".format(recon_loss))

            kl_div = kl_div_gaussian(z_mu, z_std)
            # print("KL loss {}".format(recon_loss))

            sample_loss = recon_loss + kl_div
            sample_loss = - sample_loss.detach().cpu().numpy()
            loss.append(sample_loss)

        loss = np.asarray(loss)
        sumexp = np.sum(np.exp(loss))
        # print("sum exp loss {}".format(sumexp))

        loss = np.log(sumexp) - np.log(sample_size)
        data_nll.append(loss)

    data_nll = np.asarray(data_nll)
    data_nll = - np.mean(data_nll)
    return data_nll


def reconstruction_example(model, data_loader, conditional, use_cuda):

    model.eval()
    num_class = data_loader.num_class
    img_shape = data_loader.img_shape[1:]

    x, y = next(iter(data_loader.test_loader))
    x = to_cuda(x) if use_cuda else x
    if conditional:
        y = one_hot(y, num_class)
        y = y.type(type_tfloat(use_cuda))
        x_hat, _, _ = model(x, y)
    else:
        x_hat, _, _ = model(x)

    x = x[:num_class].cpu().view(num_class * img_shape[0], img_shape[1])
    x_hat = x_hat[:num_class].cpu().view(
        num_class * img_shape[0], img_shape[1])
    comparison = torch.cat((x, x_hat), 1).view(
        num_class * img_shape[0], 2 * img_shape[1])
    return comparison


def generation_example(model, latent_size, data_loader, conditional, use_cuda):
    num_class = data_loader.num_class
    img_shape = data_loader.img_shape[1:]

    draw = randn((num_class, latent_size), use_cuda)
    if conditional:
        label = eye(num_class, use_cuda)
        sample = model.decode(draw, label).cpu().view(
            num_class, 1, img_shape[0], img_shape[1])
    else:
        sample = model.decode(draw).cpu().view(
            num_class, 1, img_shape[0], img_shape[1])

    return sample


def latentspace2d_example(model, data_loader, use_cuda):
    model.eval()
    num_x, num_y = 20, 20
    n_class = data_loader.num_class
    img_shape = data_loader.img_shape[1:]
    batch_size = data_loader.batch_size

    x_values = np.linspace(-3, -3, num_x)
    y_values = np.linspace(-3, -3, num_y)

    canvas = np.empty((num_y * img_shape[0], num_y * img_shape[1]))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            draw = torch.from_numpy(
                np.array([[np.float(xi), np.float(yi)]] * batch_size))
            draw = draw.type(type_tfloat(use_cuda))

            x_hat = model.decode(draw).cpu().detach().numpy()
            x_hat = x_hat[0].reshape(img_shape[0], img_shape[1])

            canvas[(num_x - i - 1) * img_shape[1]:(num_x - i) * img_shape[1],
                   j * img_shape[1]:(j + 1) * img_shape[1]] = x_hat

    return canvas


def latentcluster2d_example(model, data_loader, use_cuda):
    model.eval()
    centroids_x, centroids_y = [], []
    labels = []
    for _, (x, y) in enumerate(data_loader.test_loader):
        x = to_cuda(x) if use_cuda else x
        _, z, _ = model(x)
        z = z.detach().cpu().numpy()
        centroids_x.extend(z[:, 0])
        centroids_y.extend(z[:, 1])
        y = y.detach().cpu().numpy()
        labels.extend(y.flatten())

    centroids = np.vstack((np.asarray(centroids_x), np.asarray(centroids_y))).T
    return centroids, labels


def save_checkpoint(state, filename):
    torch.save(state, filename)
