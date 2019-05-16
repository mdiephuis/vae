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


# should be an iter next
def latentspace2d_example(model, data_loader, use_cuda):
    model.eval()
    for _, (x, y) in enumerate(data_loader.test_loader):
        x = to_cuda(x) if use_cuda else x
        _, z, _ = model(x)
        break
    return z.detach().cpu().numpy(), y.cpu().numpy()


def save_checkpoint(state, filename):
    torch.save(state, filename)
