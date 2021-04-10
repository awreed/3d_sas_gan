import torch.nn as nn
import torch


"""
The generator network in terms of DCGAN without batch normalization.
    nz  : input latent vector
    ngf : size of feature maps of G, 64 default.  
    nc  : number of color channels
    ngpu: number of CUDA devices available
"""
class DCGAN_G(torch.nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(DCGAN_G, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ngpu = ngpu

        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            torch.nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, int(ngf * 4 / 2),
                      kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=3, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(ngf * 4 / 2), int(ngf * 2 / 2),
                      kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(ngf * 2 / 2), ngf,
                      kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, 1,
                      kernel_size=3, stride=1, padding=2),

            torch.nn.Sigmoid()
        )

    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class DCGAN_G_3D(torch.nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(DCGAN_G_3D, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ngpu = ngpu

        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            torch.nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ngf * 8, int(ngf * 4 / 2),
                      kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=3, mode='trilinear'),
            nn.Conv3d(int(ngf * 4 / 2), int(ngf * 2 / 2),
                      kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(int(ngf * 2 / 2), ngf,
                      kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(ngf, 1,
                      kernel_size=3, stride=1, padding=3),

            torch.nn.Tanh()
        )

    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output
