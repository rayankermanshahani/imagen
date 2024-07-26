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
class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()


# ---------------- MODEL TRAINING ------------------ 


# ---------------- UTILITY FUNCTIONS ---------------

# export model weights
def save_model(model: nn.Module, file_name: str):
    torch.save(model.state_dict(), file_name)

# import model weights
def load_model(model: nn.Module, file_name: str) -> nn.Module:
    model.load_state_dict(torch.load(file_name))
    return model


# ---------------- MODEL INFERENCE ----------------- 

# ---------------- DRIVER PROGRAM ------------------ 
def main():
    print(f"using device: {device}")

    # set hyperparameters
    batch_size = 64

    # load dataset
    print("loading dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    dataset = TensorDataset(x_train, y_train.argmax(dim=1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize models
    # print("initializing models...")

    # train the vae system
    # print("training VAE system...")

    # save models

    print("\ntraining complete. models successfully exported.")

if __name__ == "__main__":
    main()


