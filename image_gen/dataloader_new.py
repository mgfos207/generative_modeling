from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as v2
from torchvision.utils import save_image
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import os
from torchvision.utils import make_grid
from discriminator import Discriminator
from generator import Generator
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm.notebook import tqdm
import torch.nn.functional as F



class AnimeFaces(Dataset):
    def __init__(self, dir_path, augmentations=None):
        self.dir_path = dir_path
        self.img_paths = os.listdir(dir_path)
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        if idx < len(self.img_paths):
            image_path = self.img_paths[idx]
            try:
                # image = cv2.imread(os.path.join(self.dir_path, image_path))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.open(os.path.join(self.dir_path, image_path))
                if self.augmentations != None:
                    image = self.augmentations(image)
                
                return image
            except:
                return self.__getitem__(idx + 1)

        # return None
class AnimeDataLoader:
    def __init__(self, img_size, batch_size, dir_path):
        self.stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        # self.augmentations = v2.Compose([
        #         v2.Resize(img_size),
        #         v2.ToTensor(),
        #         v2.Normalize(*self.stats)
        #     ])
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dataset =  ImageFolder(
            dir_path, transform=v2.Compose([
                v2.Resize(img_size),
                v2.ToTensor(),
                v2.Normalize(*self.stats)
            ])
        )
        # self.dataset = self.to_device(self.dataset)
        # self.dataset = AnimeFaces(dir_path, augmentations=self.augmentations)
        self.data_loader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: tuple(x_.to(self.device) for x_ in default_collate(x)))
    
    def denorm(self, img_tensors):
        return img_tensors * self.stats[1][0] + self.stats[0][0]

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
    def __init__(self, dl, fixed_latent):
        self.dl = dl
        self.fixed_latent = fixed_latent
        self.latent_size = 64
        self.batch_size = 20
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.discriminator = self.to_device(self.create_discriminator())
        # self.generator = self.to_device(self.create_generator())
        self.discriminator = self.to_device(self.create_discriminator())
        self.generator = self.to_device(self.create_generator())

    
    def __iter__(self):
        for b in self.dl:
            self.to_device(b)

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
        return Discriminator()
    
    def create_generator(self):
        return Generator(self.latent_size)

    def train_discriminator(self, real_images, opt_d):
        # Clear discriminator gradients
        opt_d.zero_grad()

        disc_criterion = nn.BCELoss()
        disc_criterion.cuda()

        # Pass real images through discriminator
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=self.device)
        # real_targets = real_targets.squeeze()
        # real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_loss = disc_criterion(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()
        
        # Generate fake images
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        # Pass fake images through discriminator
        fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        # Update discriminator weights
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score

    def train_generator(self, opt_g):
        # Clear generator gradients
        opt_g.zero_grad()
        
        # Generate fake images
        latent = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)
        
        # Try to fool the discriminator
        preds = self.discriminator(fake_images)
        targets = torch.ones(self.batch_size, 1, device=self.device)
        loss = F.binary_cross_entropy(preds, targets)
        
        # Update generator weights
        loss.backward()
        opt_g.step()
        
        return loss.item()

    def save_samples(self, index, latent_tensors, show=True):
        fake_images = self.generator(latent_tensors)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        # save_image(self.denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
        # print('Saving', fake_fname)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))

    def fit(self, epochs, lr, start_idx=1):
        torch.cuda.empty_cache()
        
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        
        # Create optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        for epoch in range(epochs):
            for real_images, _ in tqdm(self.dl):
                # Train discriminator
                loss_d, real_score, fake_score = self.train_discriminator(real_images, opt_d)
                # Train generator
                loss_g = self.train_generator(opt_g)
                
            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            
            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        
            # Save generated images
            self.save_samples(epoch+start_idx, self.fixed_latent, show=False)
        
        return losses_g, losses_d, real_scores, fake_scores

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