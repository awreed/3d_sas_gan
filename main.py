import torch
import argparse
import os
import torchvision
import random
import json

from models.DCGAN_G import DCGAN_G_nobn
from models.DCGAN_D import DCGAN_D_nobn
from load_data import get_data_from_pt, gradient_penalty
import matplotlib.pyplot as plt


def weights_init(input):
    classname = input.__class__.__name__
    if classname.find('Conv') != -1:
        input.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        input.weight.data.normal_(1.0, 0.02)
        input.bias.data.fill_(0)

# Define some parameters
nz = 100 # z dimension
ngf = 64 # number of generator features
ngpu = 1 # number of gpus
nc = 1 # number of image channels
ndf = 64 # number of discriminator features
batch_size = 1
lrD = .000005 # learning rate of the disc
lrG = .000005 # learning rate of the generator
beta1 = 0.5 # adam param
nepoch = 10000000
Diters = 1 # how many times to train the discimrinator for each gerantor itearion
clamp_lower = -.1 # enforcing smoothness on disc lower bound
clamp_upper = .1 # enforcing smoothness on disc upper bound

save_dir = 'results/'


device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

# Define the generator
netG = DCGAN_G_nobn(nz, ngf, nc, ngpu).to(device)
netG.apply(weights_init)

# Define the discriminator
netD = DCGAN_D_nobn(nc, ndf, ngpu).to(device)
netD.apply(weights_init)

# some constants
fixed_noise = torch.FloatTensor(torch.randn(batch_size, nz, 1, 1)).to(device)
one = torch.FloatTensor([1]).to(device)
mone = one * -1

# define the optimizers
optimizerD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

# Load in the single data sample
data = get_data_from_pt('complex_test.pt')
data_2d_slice = data[..., 50, :].squeeze().numpy()
#plt.imshow(data_2d_slice)
#plt.show()
plt.imsave(save_dir + 'gt.png', data_2d_slice)

# Here is the data tensor we will feed to the discriminator [B, C, H, W] -> [1, 1, 100, 100]
real_data = torch.from_numpy(data_2d_slice)[None, None, ...].to(device)
real_data = real_data.float() # weight type is float
gen_iters = 0
print("Starting Training Loop...")
for epoch in range(nepoch):

    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    # train the discriminator Diters times
    #if gen_iters < 25 or gen_iters % 500 == 0:
    #    Diters = 100
    #else:
    #    Diters = Diters

    j = 0
    while j < Diters:
        j += 1

        # This is part of the wasserstein loss (enforce Lipschitz smoothness on the discriminator)
        #for p in netD.parameters():
        #    p.data.clamp_(clamp_lower, clamp_upper)

        # Show the discriminator the real data
        netD.zero_grad()
        # Format batch

        errD_real = netD(real_data)
        errD_real.backward(one)

        # show the discriminator the fake data
        # Sample batch of latent vectors.
        with torch.no_grad():  # totally freeze netG
            noise = torch.randn(batch_size, nz, 1, 1).to(device)

        fake = netG(fixed_noise)
        print(fake.min(), fake.max())
        errD_fake = netD(fake.detach())
        errD_fake.backward(mone)
        # this is the wasserstein loss
        errD = errD_real - errD_fake + gradient_penalty(real_data, fake, netD)
        optimizerD.step()

    # train the generator
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()
    # in case our last batch was the tail batch of the dataset,
    # make sure we feed a full batch of noise
    errG = netD(fake)
    errG.backward(one)
    optimizerG.step()
    gen_iters += 1

    # save some stuff
    if gen_iters % 10 == 0:
        fake = netG(fixed_noise)
        fake = fake.squeeze().detach().cpu().numpy()
        plt.imsave(save_dir + 'est' + str(epoch) + '.png', fake)

    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.models, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.models, epoch))
