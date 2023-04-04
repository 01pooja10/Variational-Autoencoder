import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, inp, h=200, z=20) -> None:
        super().__init__()
        
        self.ih = nn.Linear(inp, h)
        self.mu = nn.Linear(h,z)
        self.std = nn.Linear(h,z)
        
        self.rebuild = nn.Linear(z,h)
        self.hi = nn.Linear(h,inp)
        
        self.relu = nn.ReLU()
        
        
    #qphi (z|x)
    def encoder(self, img):
        x = self.relu(self.ih(img))
        mu,std = self.mu(x), self.std(x)
        return mu, std
        
    #ptheta (x|z)
    def decoder(self, z):
        x = self.relu(self.rebuild(z))
        x = self.hi(x)
        x = torch.sigmoid(x)
        return x
        
    def forward(self, x):
        mu, std = self.encoder(x)
        
        #gaussian noise - minimizes overfitting
        e = torch.randn_like(std)
        
        #add noise to output
        z_reparam = mu + std*e
        
        x_reconst = self.decoder(z_reparam)
        
        return mu, std, x_reconst
    
    
def main():
    x = torch.randn(2,784)
    vae = VAE(784)
    out = vae(x)
    m,s,x = out
    print(m.shape, s.shape, x.shape)
    
if __name__ == "__main__":
    main()
    
