import torch
import torch.nn as nn
import torch.nn.functional as F

"""
nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2,  padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

            nn.Flatten(),
            nn.Sigmoid()

"""
# Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Simple CNN
        # self.main = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv2d(256, 512, kernel_size=4, stride=2,  padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(0.2, inplace=True),

        #     nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

        #     nn.Flatten(),
        #     nn.Sigmoid()
        # )
        self.ngpu = 1
        self.main = nn.Sequential(
        # in: 3 x 64 x 64

        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 64 x 32 x 32

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 128 x 16 x 16

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 256 x 8 x 8

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        # out: 512 x 4 x 4

        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
        # out: 1 x 1 x 1

        nn.Flatten(),
        nn.Sigmoid())
  
    def forward(self, x):
        # gpu_ids = None
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #     gpu_ids = range(self.ngpu)
        # output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        # return output.view(-1, 1)
        return self.main(x)