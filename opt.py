import argparse
from data import get_mnist, get_cifar10, get_fashion

"""
options
"""

def get_options():
    parser = argparse.ArgumentParser()
    parser = normal_options(parser)
    parser = model_options(parser)
    parser = infogan_options(parser)

    return parser.parse_args()


def normal_options(parser):

    # parameters in training
    parser.add_argument("--lrD", default=2e-4, type=float, help="learning rate of Discriminator")
    parser.add_argument("--lrG", default=2e-4, type=float, help="learning rate of Generator")
    parser.add_argument("--epochs", default=10, type=int, help="total epochs in training")
    parser.add_argument("--save-epoch-interval", default=1, type=int, help="interval of epoch for saving model")
    parser.add_argument("--cuda", default=True, type=bool, help="using cuda")
    parser.add_argument("--adam-betas", default=(0.9, 0.999), type=tuple, help="betas of adam optimizer")

    # parameters of dataset
    parser.add_argument("--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("--data-name", default="MNIST", type=str, help="dataset name")
    # parser.add_argument("--img-size", default=(64,64), type=tuple, help="output image size") # currently cannot be changed
    parser.add_argument("--nrow", default=16, type=int, help="number of rows when showing batch of images")
    parser.add_argument("--data-path", default="data", type=str, help="path of dataset")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers in dataloader")
    parser.add_argument("-inc", "--in-channels", default=3, type=int, help="number of channels of input")

    # misc 
    parser.add_argument("--test", default=False, type=bool, help="train one epoch for test")
    parser.add_argument("--board", default=True, type=bool, help="using tensorboard to record") # temporay use tensorboard as default
    
    return parser


def infogan_options(parser):
    parser.add_argument("-ncz", default=3, type=int, help="number of continuous factors")
    parser.add_argument("-ndlist", default=[10], type=list, help="number of classes of each discrete factors")
    parser.add_argument("--lambda", default=1, type=float, help="hyperparameters of mutual information")
    return parser


def model_options(parser):
    parser.add_argument("--dim-z", default=100, type=int, help="dimension of latent variable z")
    return parser


def get_traverse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True, help="full path of saved netG.pth")
    parser.add_argument("--cuda", default=True, type=bool, help="using cuda")
    parser.add_argument("--didx", type=int, default=-1, help="index of desired traversal discrete variable")
    parser.add_argument("--cidx", type=int, default=-1, help="index of desired traversal continuous variable")
    parser.add_argument("--c-range", type=tuple, default=(-2,2), help="range of continuous variable in traversal")
    parser.add_argument("--batch-size", default=100, type=int, help="batch size")
    parser.add_argument("--nrow", default=10, type=int, help="number of rows when showing batch of images")
    parser.add_argument("--out-name", default="test", type=str, help="name of output images")
    parser.add_argument("--in-channels", default=1, type=int, help="number of channels of input image")
    parser.add_argument("--seed", default=5224, type=int, help="random seed")
    parser.add_argument("--fixmode", default=False, type=bool, help="using fix mode, fix targeted variables")

    parser = model_options(parser)
    parser = infogan_options(parser)

    return parser.parse_args()


def choose_dataset(opt):
    """ choose dataset
    """
    data_name = opt.data_name
    if data_name == "MNIST":
        setattr(opt, "in_channels", 1)
        data = get_mnist(opt.data_path, opt.batch_size, opt.num_workers)
    elif data_name == "cifar10":
        setattr(opt, "in_channels", 3)
        data = get_cifar10(opt.data_path, opt.batch_size, opt.num_workers)
    elif data_name == "fashion":
        setattr(opt, "in_channels", 1)
        data = get_fashion(opt.data_path, opt.batch_size, opt.num_workers)
    else:
        raise NotImplementedError("Not implemented dataset: {}".format(data_name))
    return data


class _MetaOptions:
    """ options-like object
    """
    def __str__(self):
        return ";".join(["{}:{}".format(key,val) for key, val in self.__dict__.items()])
    
    @staticmethod
    def kws2opts(**kws):
        """ Recursively convert all keyword input to option like object.
        """
        return _MetaOptions.dict2opts(kws)

    @staticmethod
    def dict2opts(d: dict):
        """ Recursively convert dict to option like object.
        """
        o = _MetaOptions()
        def _parse(obj, dt: dict):
            for key, val in dt.items():
                if not isinstance(key, str):
                    raise AttributeError("Not allowed key in dict with type:{}".format(type(key)))
                if isinstance(val, dict):
                    t = _MetaOptions()
                    setattr(obj, key, t)
                    _parse(t, val)
                else:
                    setattr(obj, key, val)
            return obj
        return _parse(o, d)


if __name__ == "__main__":
    opt = _MetaOptions.kws2opts(name="test", lr=1e-3, epochs=20)
    print(opt.name)
    print(opt.lr, opt.epochs)
