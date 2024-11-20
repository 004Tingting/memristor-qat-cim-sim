"""MNIST loaders (downloaded once into the project's data/ folder)."""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist(root):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    val = datasets.MNIST(root, train=False, download=True, transform=tf)
    return {
        "train": DataLoader(train, batch_size=256, shuffle=True, num_workers=0),
        "val": DataLoader(val, batch_size=512, shuffle=False, num_workers=0),
    }
