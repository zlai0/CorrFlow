import os,sys
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def image_loader(path):
    image = cv2.imread(path)
    image = np.float32(image) / 255.0
    image = cv2.resize(image, (256, 256))
    return image

def rgb_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(image)

def rgb_preprocess_jitter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image = transforms.ColorJitter(0.1,0.1,0.1,0.1)(image)
    image = transforms.ToTensor()(image)
    return image

def greyscale_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lightness = image[None,:,:,0]
    processed = lightness / 50 - 1  # 0 mean
    return torch.Tensor(processed)

def quantized_color_preprocess(image, centroids):
    h, w, c = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    ab = image[:,:,1:]

    a = np.argmin(np.linalg.norm(centroids[None, :, :] - ab.reshape([-1,2])[:, None, :], axis=2),axis=1)
    # 256 256  quantized color (4bit)

    quantized_ab = a.reshape([h, w, -1])
    preprocess = transforms.ToTensor()
    return preprocess(quantized_ab)

class myImageFloder(data.Dataset):
    def __init__(self, filepath, filenames, training):

        self.refs = filenames
        self.filepath = filepath
        self.training = training
        self.p_2 = 0.1             # probability of random perturbation
        self.centroids = np.load('datas/centroids/centroids_16k_kinetics_10000samples.npy')

    def __getitem__(self, index):
        refs = self.refs[index]

        images = [image_loader(os.path.join(self.filepath, ref)) for ref in refs]

        images_quantized = [quantized_color_preprocess(ref, self.centroids) for ref in images]

        r = np.random.random()
        if r < self.p_2:
            images_rgb = [rgb_preprocess_jitter(ref) for ref in images]
        else:
            images_rgb = [rgb_preprocess(ref) for ref in images]

        return images_rgb, images_quantized

    def __len__(self):
        return len(self.refs)

