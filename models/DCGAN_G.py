import torch.nn as nn
import torch


"""
The generator network in terms of DCGAN without batch normalization.
    nz  : input latent vector
    ngf : size of feature maps of G, 64 default.  
    nc  : number of color channels
    ngpu: number of CUDA devices available
"""
class DCGAN_G_nobn(torch.nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(DCGAN_G_nobn, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ngpu = ngpu

        '''
        torch.nn.ConvTranspose2d: Applies a 2D transposed convolution operator over an input image.
        #torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, \
         padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        #bias=True: adds a learnable bias to the output.
        '''
        self.main = torch.nn.Sequential(
            # 100->512, stride=1
            torch.nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            torch.nn.ReLU(inplace=True),
            # 512->256, stride=2, padding=1
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, int(ngf * 4 / 2),
                      kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            # 256->128, stride=2, padding=1
            #torch.nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.Upsample(scale_factor=3, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(int(ngf * 4 / 2), int(ngf * 2 / 2),
                      kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(inplace=True),
            # 128->64, stride=2, padding=1
            #torch.nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 0, bias=False),
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

        """
        Forward propogation of G.
        """
        '''
        #torch.nn.parallel.data_parallel(module, inputs, device_ids=None, \
         #output_device=None, dim=0, module_kwargs=None)
         #module: the module to evaluate in parallel, self.net
         #input : inputs to the module
         #device_ids:GPU ids on which to replicate module
         #output_device:GPU location of the output Use -1 to indicate the CPU. (default: device_ids[0])
        '''
    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

