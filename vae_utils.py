import numpy as np
import torch
from nn_helpers.utils import one_hot, to_cuda, type_tfloat, randn, eye


def reconstruction_example(model, data_loader, conditional, use_cuda):

    model.eval()
    num_class = data_loader.num_class
    img_shape = data_loader.img_shape[1:]

    for _, (x, y) in enumerate(data_loader.test_loader):
        x = to_cuda(x) if use_cuda else x
        if conditional:
            y = one_hot(y, num_class)
            y = y.type(type_tfloat(use_cuda))
            x_hat, _, _ = model(x, y)
        else:
            x_hat, _, _ = model(x)
        break

    x = x[:num_class].cpu().view(num_class * img_shape[0], img_shape[1])
    x_hat = x_hat[:num_class].cpu().view(num_class * img_shape[0], img_shape[1])
    comparison = torch.cat((x, x_hat), 1).view(num_class * img_shape[0], 2 * img_shape[1])
    return comparison


def generation_example(model, latent_size, data_loader, conditional, use_cuda):
    num_class = data_loader.num_class
    img_shape = data_loader.img_shape[1:]

    draw = randn((num_class, latent_size), use_cuda)
    if conditional:
        label = eye(num_class, use_cuda)
        sample = model.decode(draw, label).cpu().view(num_class, 1, img_shape[0], img_shape[1])
    else:
        sample = model.decode(draw).cpu().view(num_class, 1, img_shape[0], img_shape[1])

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
            draw = torch.from_numpy(np.array([[np.float(xi), np.float(yi)]] * batch_size))
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
