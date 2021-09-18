import torch
import torch.nn as nn
import torch.nn.functional as F


#------------------ DCGAN Model for infoGAN -------------------------
class ModDiscriminator(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(ModDiscriminator, self).__init__()

        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        feature = self.net(x)
        return feature


class ModGenerator(nn.Module):
    def __init__(self, out_channels, latent_dim):
        super(ModGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.meta_dim = 512
        self.latent_mapping = nn.Sequential(
            nn.Linear(latent_dim, self.meta_dim * 4 * 4)
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        w = self.latent_mapping(z).view(z.size(0), -1, 4, 4)
        return self.net(w)
