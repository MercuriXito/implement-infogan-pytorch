import torch
import torch.nn as nn
import torch.nn.functional as F

from opt import get_traverse_options

"""
Utils functions in InfoGAN
"""

class NoiseGenerator:
    """ Noise Generator for the input of InfoGAN
    """
    def __init__(self, total_latent_dim: int, num_classes: list, num_con_var: int, 
            device=None):
        """
        :params:
                total_latent_dim    - total dim of all latent varibles.
                num_classes         - number of class of each discrete variables.
                num_con_var         - number of continuous variables.
        """
        self.zdim = total_latent_dim

        self.ddim = sum(num_classes) # dimesion of all discrete variables
        self.num_classes = num_classes
        self.dnum = len(num_classes)

        self.cdim = num_con_var # dimesion of all continuous variables
        self.cnum = num_con_var

        self.udim = self.zdim - self.ddim - self.cdim
        if self.udim <= 0:
            raise AttributeError("Total dimension of latent: {} is too small".format(self.zdim))

        if device is None:
            self.device = torch.device("cuda")
        else:
            self.device = device

    def random_get(self, batch_size):
        """Generate random vectors for training.
        """

        z = torch.randn((batch_size, self.udim), device=self.device)
        z_con = torch.rand((batch_size, self.cdim), device=self.device) * 2 - 1

        labels = []
        z_dis = []
        for num_class in self.num_classes:
            label_dis = torch.randint(0, num_class, (batch_size, ), device=self.device)
            labels.append(label_dis)
            z_dis.append(F.one_hot(label_dis, num_classes=num_class).type(torch.FloatTensor).to(self.device))
        z_dis = torch.cat(z_dis, dim=1)

        return [z, z_con, z_dis, labels]

    def traversal_get(
            self, batch_size, idx_con=-1, idx_dis=-1,
            z_con_range=(-1, 1), seed=5224, fix_mode=False):
        """Generate random vectors for traveral.
        """

        assert isinstance(z_con_range, (tuple, list)) and len(z_con_range) == 2

        if fix_mode:
            z = self.fix_target(batch_size, idx_con, idx_dis, z_con_range, seed)
        else:
            z = self.fix_other_traverse(batch_size, idx_con, idx_dis, z_con_range, seed)
        return z

    def fix_target(
            self, batch_size, idx_con=-1, idx_dis=-1,
            z_con_range=(-1, 1), seed=5224):
        """ traverse while fixing targeted variables
        """

        torch.random.manual_seed(seed)

        # fixed random noise
        z = torch.randn((batch_size, self.udim))
        z_con = torch.rand((batch_size, self.cdim))

        z_dis = []
        for num_class in self.num_classes:
            labels = torch.randint(0, num_class, (batch_size, ))
            z_dis.append(F.one_hot(labels, num_classes=num_class).type(torch.FloatTensor))
        z_dis = torch.cat(z_dis, dim=1)

        # traversal part
        # continuous variables
        if idx_con >= 0 and idx_con < self.cdim:
            z_change = torch.randn((1,))
            z_con[:, idx_con] = z_change

        # discrete variables
        if idx_dis >= 0 and idx_dis < self.dnum:
            num_traversal_class = self.num_classes[idx_dis]
            labels = torch.randint(0, num_class, (1, )).repeat(batch_size)
            z_change = F.one_hot(labels, num_classes=num_traversal_class).type(torch.FloatTensor)

            front_idx = sum(self.num_classes[:idx_dis])
            z_dis[:, front_idx: front_idx + num_traversal_class] = z_change

        z = torch.cat([z, z_con, z_dis], dim = 1).to(self.device)
        return z

    def fix_other_traverse(
            self, batch_size, idx_con=-1, idx_dis=-1,
            z_con_range=(-1, 1), seed=5224):
        """ traverse while fixing other unrelated variables
        """

        torch.random.manual_seed(seed)

        # fixed random noise
        z = torch.randn((1, self.udim)).repeat(batch_size, 1)
        z_con = torch.rand((1, self.cdim)).repeat(batch_size, 1)

        z_dis = []
        for num_class in self.num_classes:
            labels = torch.randint(0, num_class, (1, ))
            z_dis.append(F.one_hot(labels, num_classes=num_class).type(torch.FloatTensor))
        z_dis = torch.cat(z_dis, dim=1).repeat(batch_size, 1)

        # traversal part
        # continuous variables
        if idx_con >= 0 and idx_con < self.cdim:
            z_change = torch.linspace(z_con_range[0],z_con_range[1],batch_size)
            z_con[:, idx_con] = z_change

        # discrete variables
        if idx_dis >= 0 and idx_dis < self.dnum:
            num_traversal_class = self.num_classes[idx_dis]
            labels = torch.linspace(0, num_traversal_class, batch_size + 1).type(torch.LongTensor)
            labels = labels[:-1]
            z_change = F.one_hot(labels, num_classes=num_traversal_class).type(torch.FloatTensor)

            front_idx = sum(self.num_classes[:idx_dis])
            z_dis[:, front_idx: front_idx + num_traversal_class] = z_change

        z = torch.cat([z, z_con, z_dis], dim = 1).to(self.device)
        return z


class NormalNLLLoss:
    """Negative Log Likelihood Loss of Gaussian Distribution, ignore the constant.
    """
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, x, mu, logvar):
        """Negative Log Likelihood Loss of Gaussian Distribution, ignore the constant.

        :params:
                x           - batch input
                mu          - mean of Gaussian Distribution
                logvar      - log variance of Gaussian Distribution
        """

        loss = -0.5 * (logvar + (x - mu) ** 2 /(logvar.exp() + self.eps)).mean()
        return loss * -1


class InfoGANLoss:
    """ Loss of InfoGAN
    """
    def __init__(self, beta: float = 1.0, device=None):
        self.beta = beta
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        self.con_criterion = NormalNLLLoss()
        if device is None:
            self.device = torch.device("cuda")
        self.adv_labels = torch.tensor(0, dtype=float).to(self.device).requires_grad_(False)

    def get_adv_loss(self, adv_out, adv_label):
        """ Adversarial loss
        """
        return self.adv_criterion(adv_out, adv_label)

    def get_mi_loss(self, mi_out, dis_labels:list, con_z):
        """ Negative of mutual information loss
        """
        dis_outs, con_out = mi_out
        # mi of discrete c_i
        dis_losses = torch.tensor(0, dtype=float, device=self.device)
        for i, (dis_out, dis_label) in enumerate(zip(dis_outs, dis_labels)):
            dis_loss = self.dis_criterion(dis_out, dis_label)
            dis_losses += dis_loss
        dis_losses = dis_losses / (i + 1)

        # mi of continuous c_i
        mean_out, logvar_out = con_out
        con_loss = self.con_criterion(con_z, mean_out, logvar_out)

        return self.beta * (dis_losses + con_loss)

    def __call__(self, out_d, adv_label, dis_labels: list, con_z):
        """
        :params:
            out_d               (tuple)     - complete output of Discriminator Network.
            adv_label           (tensor)    - fake or true samples
            dis_labels          (list)      - ground true labels of all discrete variables
            con_z               (tensor)    - latent value of continuous variables
        """

        adv_out, dis_outs, con_out = out_d
        adv_loss = self.adv_criterion(adv_out, adv_label)

        # mi of discrete c_i
        dis_losses = torch.tensor(0, dtype=float, device=self.device)
        for i, (dis_out, dis_label) in enumerate(zip(dis_outs, dis_labels)):
            dis_loss = self.dis_criterion(dis_out, dis_label)
            dis_losses += dis_loss
        dis_losses = dis_losses / (i + 1)

        # mi of continuous c_i
        mean_out, logvar_out = con_out
        con_loss = self.con_criterion(con_z, mean_out, logvar_out)

        return adv_loss + self.beta * ( dis_losses + con_loss).mean()


def traverse():
    opt = get_traverse_options()

    from models.dcgan import Generator
    from utils.misc import TensorImageUtils

    utiler = TensorImageUtils()
    in_channels = opt.in_channels
    if opt.data_name == "MNIST":
        in_channels = 1
    dim_z = opt.dim_z
    num_classes = opt.ndlist
    num_categorical_variables = len(num_classes)
    num_continuous_variables = opt.ncz

    device = torch.device("cuda:0")
    netG = Generator(in_channels, dim_z)
    netG.cuda()

    netG.load_state_dict(torch.load(opt.model_path))

    g = NoiseGenerator(dim_z, num_classes, num_continuous_variables)
    z = g.traversal_get(opt.batch_size, opt.cidx, opt.didx, opt.c_range, opt.seed, opt.fixmode)
    # z = g.random_get(100)
    # z = torch.cat(z[:3], dim=-1)
    print(z.size())

    x = netG(z)
    output_name = "{}.png".format(opt.out_name)
    utiler.save_images(x, output_name, nrow=opt.nrow)
    print("Save traversal image in {}".format(output_name))


if __name__ == "__main__":
    traverse()
