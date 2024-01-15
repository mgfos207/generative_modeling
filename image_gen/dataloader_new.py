from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
import torch.nn as nn
import os
from torchvision.utils import make_grid
from discriminator import Discriminator
from generator import Generator
import matplotlib.pyplot as plt
import cv2



class AnimeFaces(Dataset):
    def __init__(self, dir_path, augmentations=None):
        self.dir_path = dir_path
        self.img_paths = os.listdir(dir_path)
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = cv2.imread(os.path.join(self.dir_path, image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.augmentations != None:
                image = self.augmentations(image)
            
            return image
        except:
            return self.__getitem__(idx + 1)

class AnimeDataLoader:
    def __init__(self, img_size, batch_size, dir_path):
        self.stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        self.augmentations = T.Compose([
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(*self.stats)
            ])
        self.dataset = AnimeFaces(dir_path, augmentations=self.augmentations)
        self.data_loader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size)
    
    def denorm(self, img_tensors):
        return img_tensors * self.stats[1][0] + stats[0][0]

    def show_images(self, images, nmax=64):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(self.denorm(images.detach()[:nmax]), nrow=8).permute(1,2,0))

    def show_batch(self, dl, nmax=64):
        for images, _ in dl:
            self.show_images(images, nmax)
            break

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, self.device) for x in data]
        return data.to(self.device, non_blocking=True)

class AnimeGAN:
    def __init__(self, dl):
        self.dl = dl
        self.latent_size = 128
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.discriminator = self.to_device(self.create_discriminator())
        # self.generator = self.to_device(self.create_generator())
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()

    
    def __iter__(self):
        for b in self.dl:
            self.to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, self.device) for x in data]
        return data.to(self.device, non_blocking=True)

    def denorm(self, img_tensors):
        return img_tensors * stats[1][0] + stats[0][0]
    
    def show_images(self, images, nmax=64):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(self.denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

    def show_batch(self, dl, nmax=64):
        for images, _ in dl:
            self.show_images(images, nmax)
            break
    
    def create_discriminator(self):
        return Discriminator
        # return nn.Sequential(
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
    
    def create_generator(self):
        return Generator
        # return nn.Sequential(
        #     # in: latent_size x 1 x 1

        #     nn.ConvTranspose2d(self.latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(True),
        #     # out: 512 x 4 x 4

        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(True),
        #     # out: 256 x 8 x 8

        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     # out: 128 x 16 x 16

        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     # out: 64 x 32 x 32

        #     nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.Tanh()
        #     # out: 3 x 64 x 64
        # )



# class AnimeDataLoader:
#     def __init__(self, image_size, batch_size, data_dir ):
#         self.img_size = image_size
#         self.batch_size = batch_size
#         self.data_dir = data_dir
#         self.stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
#         self.train_ds = ImageFolder(
#             data_dir, transform=T.Compose([
#                 T.Resize(image_size),
#                 T.ToTensor(),
#                 T.Normalize(*self.stats)
#             ])
#         )
#         self.train_dl = DataLoader(self.train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     def __iter__(self):
#         for b in self.train_dl:
#             self.device(b, )
    
#     def denorm(self, img_tensors):
#         return img_tensors * self.stats[1][0] + stats[0][0]

#     def to_device(self, data):
#         if isinstance(data, (list, tuple)):
#             return [self.to_device(x, self.device) for x in data]
#         return data.to(self.device, non_blocking=True)
    
#     def show_images(self, images, nmax=64):
#         fig, ax = plt.subplots(figsize=(8, 8))
#         ax.set_xticks([]); ax.set_yticks([])
#         ax.imshow(make_grid(self.denorm(images.detach()[:nmax]), nrow=8).permute(1,2,0))

#     def show_batch(self, dl, nmax=64):
#         for images, _ in dl:
#             self.show_images(images, nmax)
#             break


image_size = 200
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)