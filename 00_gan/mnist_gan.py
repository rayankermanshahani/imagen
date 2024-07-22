#!/usr/bin/env python3 

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from tqdm import tqdm
from typing import Tuple

torch.set_printoptions(sci_mode=False)

# detect and set pytorch target device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ---------------- DATA PREPARATION ---------------- 

# load mnist dataset sourced from: https://yann.lecun.com/exdb/mnist/
def load(file_path: str):
    abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    try: 
        with open(abs_file_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    except IOError:
        print(f"error reading file: {file_path}")

    return data

def load_mnist():
    x_train = load("../data/mnist/train-images-idx3-ubyte")[0x10:].reshape((-1, 1, 28, 28))
    y_train = load("../data/mnist/train-labels-idx1-ubyte")[8:]
    x_test = load("../data/mnist/t10k-images-idx3-ubyte")[0x10:].reshape((-1, 1, 28, 28))
    y_test = load("../data/mnist/t10k-labels-idx1-ubyte")[8:]

    # TODO: some redundancy regarding dtypes and testing sets starts here, fix later
    x_train = torch.from_numpy(x_train.copy()).to(device).bfloat16() / 255.0
    y_train = F.one_hot(torch.from_numpy(y_train.copy()).to(device).long()).bfloat16()
    x_test = torch.from_numpy(x_test.copy()).to(device).bfloat16() / 255.0
    y_test = torch.from_numpy(y_test.copy()).to(device)

    return x_train, y_train, x_test, y_test

# ---------------- NETWORK ARCHITECTURE ------------ 

class Generator(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(in_features=latent_dim + num_classes, out_features=256), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024), 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=28*28), 
            nn.Tanh()
        )

    def forward(self, z: Tensor, labels: Tensor):
        label_emb = self.label_emb(labels)
        x = torch.cat([z, label_emb], dim=1)

        return self.model(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(in_features=28*28 + num_classes, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor, labels: Tensor):
        x = x.view(x.size(0), -1)
        label_emb = self.label_emb(labels)
        x = torch.cat([x, label_emb], dim=1)

        return self.model(x)


# ---------------- MODEL TRAINING ------------------ 

def train_gan(generator: Generator, discriminator: Discriminator, dataloader: DataLoader, num_epochs: int, latent_dim: int, num_classes: int) -> Tuple[Generator, Discriminator]: 
    criterion = nn.BCELoss()
    optim_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (real_imgs, labels) in enumerate(tqdm(dataloader)):
            batch_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)

            # train discriminator
            optim_d.zero_grad()

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z, labels)

            real_loss = criterion(discriminator(real_imgs, labels), torch.ones(batch_size, 1).to(device))
            fake_loss = criterion(discriminator(fake_imgs.detach(), labels), torch.zeros(batch_size, 1).to(device))
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optim_d.step()


            # train generator
            optim_g.zero_grad()

            gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z, gen_labels)

            g_loss = criterion(discriminator(gen_imgs, gen_labels), torch.ones(batch_size, 1).to(device))
            g_loss.backward()
            optim_g.step()
        print(f"epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    return generator, discriminator

# ---------------- UTILITY FUNCTIONS ---------------

# export model weights
def save_model(model: nn.Module, file_name: str):
    torch.save(model.state_dict(), file_name)

# import model weights
def load_model(model: nn.Module, file_name: str) -> nn.Module:
    model.load_state_dict(torch.load(file_name))
    return model


# ---------------- MODEL INFERENCE ----------------- 

def generate_images(generator: Generator, num_images: int, latent_dim: int, num_classes: int) -> Tuple[Tensor, Tensor]:
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        labels = torch.randint(0, num_classes, (num_images,)).to(device)
        generated_imgs = generator(z, labels)

    return generated_imgs, labels


# ---------------- DRIVER PROGRAM ------------------ 
def main():
    print(f"using device: {device}")

    # set hyperparameters
    latent_dim = 100
    num_classes = 10
    batch_size = 64
    num_epochs = 50

    # load dataset
    print("loading dataset...")
    x_train, y_train, _, _ = load_mnist()
    dataset = TensorDataset(x_train, y_train.argmax(dim=1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize models
    print("initializing models...")
    generator = Generator(latent_dim, num_classes).to(device)
    discriminator = Discriminator(num_classes).to(device)

    # train the gan system
    print("training GAN system...")
    generator, discriminator = train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, num_classes)

    # save models
    g_weights_file = "mnist_generator.pth" 
    d_weights_file = "mnist_discriminator.pth" 
    print(f"exporting generator net's weights to {g_weights_file} and discriminator net's weights to {d_weights_file}...")
    save_model(generator, g_weights_file)
    save_model(discriminator, d_weights_file)

    print("\ntraining complete. models successfully exported.")

if __name__ == "__main__":
    main()


