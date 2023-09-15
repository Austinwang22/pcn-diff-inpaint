import torch
from sde import *
from network import *
from train import *
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam
from sampling import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")

sde = VPSDE(1000)

if device.type == "cpu":
    score_model = ScoreNet(sde=sde, device=device)
else:
    score_model = torch.nn.DataParallel(ScoreNet(sde=sde, device=device), device_ids=[0])

num_epochs = 5
batch_size = 32
lr = 1e-4

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

optimizer = Adam(score_model.parameters(), lr=lr)
train(score_model, sde, data_loader, optimizer, device, num_epochs, 
      file_in='ckpts/mnist_vp.pth', file_out='ckpts/mnist_vp.pth')