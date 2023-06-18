import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import VAE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#device = 'cpu'
device = torch.device("cuda")
lr = 3e-4
inp = 784
epochs = 1
losses = []
bs = 64
h = 200
z = 20

download = True
data = datasets.MNIST(root='data/', train=True, transform = transforms.ToTensor(), download=download)
dl = DataLoader(data, batch_size = bs, shuffle = True)

model = VAE(inp, h, z).to(device)
path = "path/to/model/weights.pt"
loaded = torch.load(path)
model.load_state_dict(loaded['model'])


def inference(digit, egs=1):
    imgs = []
    idx = 0
    for x,y in data:
        if y == idx:
            imgs.append(x)
            idx+=1
        if idx==10:
            break
    encode = []
    
    for d in range(10):
        with torch.no_grad():
            mean,stddev = model.encoder(imgs[d].view(1,784))
        encode.append((mean,stddev))
        
    m,s = encode[digit]
    for eg in range(egs):
        epsilon = torch.randn_like(s)
        z_vector = m + s*epsilon
        out = model.decoder(z_vector)
        out = out.view(-1,1,28,28)
        save_image(out,f"new{digit}.png")
        
for idx in range(10):
    inference(idx, 2)
