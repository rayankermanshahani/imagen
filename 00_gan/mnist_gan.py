#!/usr/bin/env python3

import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_printoptions(sci_mode=False)

class GeneratorNet(nn.Module):
    def __init__(self, z_dim: int):
        super(GeneratorNet, self).__init__()


        


# load mnist dataset sourced from: https://yann.lecun.com/exdb/mnist/
def load(file_path: str):
    abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    try: 
        with open(abs_file_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    except IOError:
        print(f"error reading file: {file_path}")
    return data


def main():
    # detect torch device
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(type(device))
    print(device)

    x_train = load("../data/mnist/train-images-idx3-ubyte")[0x10:].reshape((-1, 1, 28, 28))
    y_train = load("../data/mnist/train-labels-idx1-ubyte")[8:]
    x_test = load("../data/mnist/t10k-images-idx3-ubyte")[0x10:].reshape((-1, 1, 28, 28))
    y_test = load("../data/mnist/t10k-labels-idx1-ubyte")[8:]

    x_train = torch.from_numpy(x_train.copy()).to(device).bfloat16() / 255.0
    y_train = F.one_hot(torch.from_numpy(y_train.copy()).to(device).long()).bfloat16()
    x_test = torch.from_numpy(x_test.copy()).to(device).bfloat16() / 255.0
    y_test = torch.from_numpy(y_test.copy()).to(device)

    print("x train: ", x_train.dtype, x_train.shape)
    print("y train: ", y_train.dtype, y_train.shape)
    print("x test: ", x_test.dtype, x_test.shape)
    print("y test: ", y_test.dtype, y_test.shape)

    x_dummy = x_train[0]


if __name__ == "__main__":
    main()


