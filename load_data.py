import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad as torch_grad
from torch.autograd import Variable


def drc_isaac_method(img, med, des_med):
    free_parameter = (des_med - med * des_med)/(med - med * des_med)
    return (img * free_parameter)/(free_parameter * img - img + 1)

def get_grid(slices, cols=5):
    [num_slices, h, w] = slices.shape

    if num_slices < cols:
        raise AssertionError('More columns than frames')

    if num_slices % cols != 0:
        raise AssertionError('number of slices must be divisible by number of columns')

    rows = int(num_slices/cols)
    image_grid = np.zeros((rows * h, cols * w))
    count = 0

    for row in range(0, rows):
        for col in range(0, cols):
            image_grid[row*h:(row+1)*h, col*w:(col+1)*w] = slices[count, ...]
            count = count + 1

    return image_grid

def get_slices(vol, axis=0):
    assert len(vol.shape) == 3, "input should be size [h, w, d]"
    [h, w, d] = vol.shape

    l = [0, 1, 2]

    assert axis in l, "axis must be either 0, 1, or 2"

    l.remove(axis)

    #allocate memory
    slices = np.zeros((vol.shape[axis], vol.shape[l[0]], vol.shape[l[1]]))

    for i in range(vol.shape[axis]):
        if axis == 0:
            slices[i, ...] = vol[i, ...]
        if axis == 1:
            slices[i, ...] = vol[:, i, :]
        if axis == 2:
            slices[i, ...] = vol[..., i]

    return slices


def get_data_from_pt(file):
    input = torch.load(file).squeeze()

    # take the absolute value of the complex data
    complex_slice = np.vectorize(complex)(input[0, ...], input[1, ...])
    abs_input = np.abs(complex_slice)

    # log normalize it
    eps = 1e-8
    log_input = 20 * np.log10(abs_input + eps)
    log_input = (log_input - log_input.min()) / (log_input.max() - log_input.min())

    # dynamic range compress it
    drc_input = drc_isaac_method(log_input.ravel(), np.median(log_input.ravel()), 0.2)
    drc_input = np.reshape(drc_input, (100, 100, 100))

    # return as shape [1, 1, H, W, D]
    return torch.from_numpy(drc_input)[None, None, ...]

def gradient_penalty(real_data, generated_data, disc, gp_weight=10):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    #interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()
    interpolated = Variable(interpolated, requires_grad=True)

    # Calculate probability of interpolated examples
    prob_interpolated = disc(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()

"""
input = torch.load('complex_test.pt').squeeze()

print(type(input))
print(input.shape)

print(input.min(), input.max())


complex_slice = np.vectorize(complex)(input[0, ...], input[1, ...])

print(complex_slice.min(), complex_slice.max())

abs_input = np.abs(complex_slice)
#abs_input = (abs_input - abs_input.min())/(abs_input.max() - abs_input.min())
eps = 1e-8
log_input = 20*np.log10(abs_input + eps)
#log_input = abs_input
log_input = (log_input - log_input.min())/(log_input.max() - log_input.min())


#plt.figure()
#plt.hist(abs_input.ravel(), 100)

drc_input = drc_isaac_method(log_input.ravel(), np.median(log_input.ravel()), 0.2)
drc_input = np.reshape(drc_input, (100, 100, 100))

#plt.figure()
#plt.hist(drc_input.ravel(), 100)

plt.show()

#logA_norm = (log_input - np.nanmin(log_input)) / (np.nanmax(log_input) - np.nanmin(log_input))

#logA_255 = 255 * logA_norm

grid = get_grid(get_slices(drc_input, axis=1), cols=10)

plt.imshow(grid)
plt.show()
"""

