import os
import torch
from tqdm import tqdm
import cv2
import numpy as np

from opt import choose_dataset, get_traverse_options
from utils.misc import de_norm
from utils.fid_score import calculate_fid_given_paths
from info_utils import NoiseGenerator
from models import Generator


class FIDEvaluator:
    def __init__(self, N=5000, batch_size=32, tmp_path='outputs/eval'):
        self.N = N
        self.path = tmp_path
        self.fake_path = os.path.join(self.path, "fake")
        self.true_path = os.path.join(self.path, "true")
        self.batch_size = batch_size
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.fake_path, exist_ok=True)
        os.makedirs(self.true_path, exist_ok=True)

    def generate_fake(self, model, noise_generator):
        num_imgs = 0
        for i in tqdm(range(self.N // self.batch_size + 1)):
            if num_imgs >= self.N:
                break

            bs = min(self.batch_size, self.N - num_imgs)
            z, zc, zd, _ = noise_generator.random_get(batch_size=self.batch_size)
            z = torch.cat([z, zc, zd], dim=1)
            with torch.no_grad():
                images = model(z)
            img_names = [f"fake{num_imgs+j:08}.png" for j in range(len(images))]
            num_imgs += bs

            # postprocess and save
            images = de_norm(images)
            for image, name in zip(images, img_names):
                image = image.detach().cpu().numpy()
                image = image.transpose(1, 2, 0)
                image = np.clip(image * 255.0, 0, 255.0).astype(np.uint8)
                path = os.path.join(self.fake_path, name)
                cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def generate_true(self, data):
        pointer = 0
        for images, _ in tqdm(data):
            bs = min(self.N - pointer, len(images))
            images = images[:bs]

            # postprocess and save
            images = de_norm(images)
            img_names = [f"true{pointer+j:08}.png" for j in range(len(images))]
            for image, name in zip(images, img_names):
                image = image.detach().cpu().numpy()
                image = image.transpose(1, 2, 0)
                image = np.clip(image * 255.0, 0, 255.0).astype(np.uint8)
                path = os.path.join(self.true_path, name)
                cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            pointer += bs
            if pointer >= self.N:
                break

    def evaluate(self, model, noise_generator, data, device):
        print("Generating true data.")
        self.generate_true(data)
        print("Generating fake data.")
        self.generate_fake(model, noise_generator)

        paths = [self.fake_path, self.true_path]
        print("Evaluating FID Score.")
        fid_value = calculate_fid_given_paths(paths,
                                              self.batch_size,
                                              device,
                                              2048,  # defaults
                                              8)
        print('FID: ', fid_value)
        return fid_value


# start evaluation.
opt = get_traverse_options()
evaluator = FIDEvaluator(N=5000, batch_size=opt.batch_size)

# load data
data = choose_dataset(opt)
if opt.data_name == "MNIST" or opt.data_name == "fashion":
    opt.in_channels = 1

# build and load model
netG = Generator(opt.in_channels, opt.dim_z)
netG.cuda()
netG.load_state_dict(torch.load(opt.model_path))
noiseG = NoiseGenerator(opt.dim_z, opt.ndlist, opt.ncz)
device = torch.device(
    "cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

evaluator.evaluate(netG, noiseG, data, device)
