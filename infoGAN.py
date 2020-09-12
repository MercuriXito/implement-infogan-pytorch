import time, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

notebook = False
if notebook:
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from models import Discriminator, Generator, QHead, DHead
from info_utils import NoiseGenerator, InfoGANLoss

from opt import get_options, choose_dataset
opt = get_options()

#------------------ configuratoin -------------------------
from utils import test_and_add_postfix_dir, test_and_make_dir, currentTime, \
    TensorImageUtils, save_model
from data import get_mnist, get_cifar10, get_fashion

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

# persistence related parameters
save_path = test_and_add_postfix_dir("tmp" + os.sep + "save_" + currentTime()) 
test_and_make_dir(save_path)
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
headD = DHead()
headQ = QHead(num_clsses, num_continuous_variables)

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

step = 1
print("Start Training, using {}".format(device))
starttime = time.clock()
for epoch in range(epochs):
    for i, batch in enumerate(tqdm(data)):
        images, _ = batch
        if use_cuda:
            images = images.cuda()

        batch_size = images.size(0)

        # ======================== Update Discriminator ===========================
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
        writer.add_scalar("lossD", lossD.item(), step)

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

        writer.add_scalar("lossG", lossG.item(), step)
        writer.add_scalar("lossG_adv", lossG_adv.item(), step)
        writer.add_scalar("lossG_mi", lossG_mi.item(), step)
        step += 1

    if (epoch + 1) % save_epoch_interval == 0 or epoch == epochs - 1:
        save_model(netG, save_path, "netG.pt")
        save_model(netD, save_path, "netD.pt")
        utiler.save_images(images, "real_{}.png".format(epoch+1), nrow=nrow)
        utiler.save_images(fake, "fake_{}.png".format(epoch+1), nrow=nrow)
        
endtime = time.clock()
consume_time = endtime - starttime
print("Training Complete, Using %d min %d s" %(consume_time // 60,consume_time % 60))
