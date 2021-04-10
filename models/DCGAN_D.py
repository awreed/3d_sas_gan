import torch


"""
The discriminator network in terms of DCGAN without batch normalization.
    nc  : number of color channels
    ndf : size of feature maps of D, 64 default
    ngpu: number of CUDA devices available
"""

class DCGAN_D(torch.nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(DCGAN_D, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu

        self.main = torch.nn.Sequential(
            #3->64, stride=2, padding=1
            torch.nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #64->128, stride=2, padding=1
            torch.nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #128->256, stride=2, padding=1
            torch.nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #256->512, stride=2, padding=1
            torch.nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            #512->1, stride=1, padding=0
            torch.nn.Conv2d(self.ndf * 8, 1, 6, 1, 0, bias=False)
        )


    def forward(self, input):
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu >1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)

class DCGAN_D_3D(torch.nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(DCGAN_D_3D, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu

        self.main = torch.nn.Sequential(
            torch.nn.Conv3d(self.nc, self.ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv3d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv3d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv3d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv3d(self.ndf * 8, 1, 6, 1, 0, bias=False)
        )


    def forward(self, input):
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        if input.is_cuda and self.ngpu >1:
            output = torch.nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)

