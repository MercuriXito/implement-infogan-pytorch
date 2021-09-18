import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
import torchvision.transforms as T


available_datasets = ('MNIST', 'CIFAR10', 'FASHION', 'SVHN', 'CELEBA')


def get_mnist(path, batch_size, num_workers):

    data = MNIST(path, train=True, transform=T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        # T.Normalize([0,5], [0.5])
        T.Normalize((0.5,),(0.5,))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_cifar10(path, batch_size, num_workers):

    data = CIFAR10(path, train=True, transform=T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_fashion(path, batch_size, num_workers):

    data = FashionMNIST(path, train=True, transform=T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5,),(0.5,))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_svhn(path, batch_size, num_workers):

    data = SVHN(path, split="train", transform=T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)


class UnlabeledCelebA(Dataset):
    """ CelebA Dataset without labels
    """
    def __init__(self, root, transform=None, compatible=True):
        self.images_path = os.path.join(self.root, "images")
        self.filenames = os.listdir(self.images_path)
        self.len = len(self.filenames)
        self.transform = transform
        self.compatible = compatible # compatible with training method using labels

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        name = self.filenames[idx]
        image = Image.open(os.path.join(self.images_path, name))

        if self.transform:
            image = self.transform(image)
        if self.compatible:
            return image, 0
        return image


def get_unlabeled_celebA(path, batch_size, num_workers):

    data = UnlabeledCelebA(path, transform=T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]))

    return DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
