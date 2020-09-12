from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms as T 


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
