import torch
import torch.nn as nn
import torch.nn.functional as F


#------------------ DCGAN Model for infoGAN -------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Discriminator, self).__init__()

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


class Generator(nn.Module):
    def __init__(self, out_channels, latent_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
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
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


# G and Discriminator are updated together
class DHead(nn.Module):
    def __init__(self, in_channels):
        super(DHead, self).__init__()
        self.dNet = nn.Conv2d(in_channels, 1, 4, 1, 0)

    def forward(self, feature):
        return self.dNet(feature).view(feature.size(0), -1)


# Q and Generator are updated together
class QHead(nn.Module):
    def __init__(self, in_channels, num_clsses: list, num_con_var: int):
        super(QHead, self).__init__()
        # Multiple parallel network for discrete variables
        self.catNets = nn.ModuleList([
            nn.Conv2d(in_channels, clss, 4, 1, 0) for clss in num_clsses
        ])

        # output mean and variance of each factors
        self.meanNet = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,2),
            nn.Conv2d(64, num_con_var, 4, 1, 0),
        )

        self.logvarNet = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0,2),
            nn.Conv2d(64, num_con_var, 4, 1, 0),
        )

    def forward(self, feature):

        bs = feature.size(0)
        dis_out = []
        for cNet in self.catNets:
            dis_out.append(cNet(feature).view(bs, -1))
        logvar_out = self.logvarNet(feature).view(bs, -1)
        mean_out = self.meanNet(feature).view(bs, -1)
        return [dis_out, (mean_out, logvar_out)]
