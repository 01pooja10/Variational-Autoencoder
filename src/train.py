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

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda")
lr = 3e-4
inp = 784
epochs = 10
losses = []
bs = 64
h = 200
z = 20

data = datasets.MNIST(root='data/', train=True, transform = transforms.ToTensor(), download=False)

dl = DataLoader(data, batch_size = bs, shuffle = True)
model = VAE(inp, h, z).to(device)
opt = torch.optim.Adam(model.parameters(), lr)
loss = nn.BCELoss(reduction="mean")

for e in range(epochs):
    loop = tqdm(enumerate(dl))
    for i, (x,_) in loop:
        x = x.to(device).view(x.shape[0],inp)
        out = model(x)
        mu, std, x_rec = out
        
        #losses
        rloss = loss(x_rec, x)
        KLdiv = -torch.sum(1+torch.log(std.pow(2)) - mu.pow(2) - std.pow(2))
        
        final_loss = rloss+KLdiv
        opt.zero_grad()
        final_loss.backward()
        opt.step()
        losses.append(final_loss)
    pprint("Epoch: ",e, "Loss value:",final_loss)


model_data = {'model': model.state_dict(),'optimizer': opt.state_dict(),'loss': losses}
torch.save(model_data,'vae_weights.pth')
        
