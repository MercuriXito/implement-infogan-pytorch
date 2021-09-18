import os
import time
from tqdm import tqdm

import torch
import numpy as np
import random
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import Discriminator, Generator, QHead, DHead
from info_utils import NoiseGenerator, InfoGANLoss
from opt import get_options, choose_dataset
from utils.misc import save_opt, TensorImageUtils, save_model

opt = get_options()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)
save_opt(save_path, opt)

#------------------ configuratoin -------------------------
# fix seed
seed = opt.seed
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# infoGAN related configuration
num_clsses = opt.ndlist
num_categorical_variables = len(num_clsses)
num_continuous_variables = opt.ncz

# training related configuration
use_cuda = opt.cuda
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
lr_D = opt.lrD
lr_G = opt.lrG
epochs = opt.epochs
save_epoch_interval = opt.save_epoch_interval
train_D_iter = opt.train_D_iter

# persistence related parameters
writer = SummaryWriter(save_path)
utiler = TensorImageUtils(save_path)
nrow = opt.nrow

# dataset
data = choose_dataset(opt)
in_channels = opt.in_channels

# models
dim_z = opt.dim_z
netD = Discriminator(in_channels, dim_z)
netG = Generator(in_channels, dim_z)
headD = DHead(512)
headQ = QHead(512, num_clsses, num_continuous_variables)

if use_cuda:
    netD.cuda()
    netG.cuda()
    headD.cuda()
    headQ.cuda()

# training config
optimizer_D = optim.Adam([
        {"params": netD.parameters()},
        {"params": headD.parameters()}
    ], lr=lr_D, betas=opt.adam_betas)

optimizer_G = optim.Adam([
        {"params": netG.parameters()},
        {"params": headQ.parameters()}
    ], lr=lr_G, betas=opt.adam_betas)

beta = 1
infoGAN_criterion = InfoGANLoss(beta)
noiseG = NoiseGenerator(dim_z, num_clsses, num_continuous_variables, device=device)

#------------------ Training -------------------------
tensor_true = torch.tensor([1], dtype=torch.float, device=device)
tensor_fake = torch.tensor([0], dtype=torch.float, device=device)

Dstep = 1
Gstep = 1
print("Start Training, using {}".format(device))
starttime = time.process_time()
for epoch in range(epochs):
    for i, batch in enumerate(tqdm(data)):
        images, _ = batch
        if use_cuda:
            images = images.cuda()

        batch_size = images.size(0)

        # ======================== Update Discriminator ===========================
        for i in range(train_D_iter):
            optimizer_D.zero_grad()

            # noise
            zu, zc, zd, zd_labels = noiseG.random_get(batch_size)
            z = torch.cat([zu, zc, zd], dim=1)

            fake = netG(z)
            out_fake = headD(netD(fake))
            labels_fake = tensor_fake.expand_as(out_fake)
            lossD_fake = infoGAN_criterion.get_adv_loss(out_fake, labels_fake)

            out_true = headD(netD(images))
            labels_true = tensor_true.expand_as(out_true)
            lossD_true = infoGAN_criterion.get_adv_loss(out_true, labels_true)
            lossD = lossD_fake + lossD_true

            lossD.backward()
            optimizer_D.step()
            writer.add_scalar("lossD", lossD.item(), Dstep)
            Dstep += 1

        # ======================== Update Generator ===========================
        optimizer_G.zero_grad()

        zu, zc, zd, zd_labels = noiseG.random_get(batch_size)
        z = torch.cat([zu, zc, zd], dim=1)

        fake = netG(z)
        out_feature = netD(fake)
        out_adv = headD(out_feature)
        out_mi = headQ(out_feature)

        labels_true = tensor_true.expand_as(out_adv)
        lossG_adv = infoGAN_criterion.get_adv_loss(out_adv, labels_true)
        lossG_mi = infoGAN_criterion.get_mi_loss(out_mi, zd_labels, zc)
        lossG = lossG_adv + lossG_mi

        lossG.backward()
        optimizer_G.step()

        writer.add_scalar("lossG", lossG.item(), Gstep)
        writer.add_scalar("lossG_adv", lossG_adv.item(), Gstep)
        writer.add_scalar("lossG_mi", lossG_mi.item(), Gstep)
        Gstep += 1

    if (epoch + 1) % save_epoch_interval == 0 or epoch == epochs - 1:
        torch.save(netG.state_dict(), os.path.join(save_path, "netG.pt"))
        torch.save(netD.state_dict(), os.path.join(save_path, "netD.pt"))
        utiler.save_images(images, "real_{}.png".format(epoch+1), nrow=nrow)
        utiler.save_images(fake, "fake_{}.png".format(epoch+1), nrow=nrow)

endtime = time.process_time()
consume_time = endtime - starttime
print("Training Complete, Using %d min %d s" %(consume_time // 60,consume_time % 60))
