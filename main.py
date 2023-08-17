import torch
from pcn_diffusion import *
from sde import *
from network import *
from train import *
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import LSUN
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from sample import *
import matplotlib.pyplot as plt
from torchvision.utils import *


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
print(f"Using {device} device")

sde = VPSDE(1000)

if device.type == "cpu":
    score_model = ScoreNet(sde=sde, device=device)
else:
    score_model = torch.nn.DataParallel(ScoreNet(sde=sde, device=device), device_ids=[0])

num_epochs = 5
batch_size = 32
lr = 1e-4

data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                     transforms.ToTensor()])

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
# dataset = CIFAR10('.', train=True, transform=data_transform, download=True)
# dataset = LSUN('.', classes=['church_outdoor_train'], transform=data_transform)
# dataset = CelebA('.', transform=data_transform, download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

H = torch.eye(28).to(device).float()
H[14][14] = 0.
H[15][15] = 0.

H2 = torch.eye(28).to(device).float()
for i in range(14):
    H2[i][i] = 0.

img, y = dataset.__getitem__(0)
img = img[0]
img = img.clamp(0., 1.)
# plt.imshow(img, vmin=0., vmax=1.)
# plt.savefig('x.png')

std_y = 0.1

y = torch.matmul(H, img.to(device)) + torch.normal(torch.zeros_like(img), std_y * torch.eye(28)).to(device)
y2 = torch.matmul(H2, img.to(device)) + torch.normal(torch.zeros_like(img), std_y * torch.eye(28)).to(device)

y_use = y2
H_use = H2

np.savetxt('y.txt', y.cpu().numpy())

plt.imshow(y.cpu(), vmin=0., vmax=1.)
plt.savefig('y.png')

tvu.save_image(y2.cpu(), 'y2.png')

optimizer = Adam(score_model.parameters(), lr=lr)

train(score_model, sde, data_loader, optimizer, device, num_epochs, 
      file_in="checkpoints/mnist_vp.pth", file_out="checkpoints/mnist_vp.pth")
# train(score_model, sde, data_loader, optimizer, device, num_epochs)

num_samples = 1

num_mcmc_steps = 2550
burn_in_period = 2500

start_sample = torch.randn((num_samples, 1, 28, 28)).to(device)
# start_sample = torch.tensor([[np.loadtxt('bin/noisy_sample.txt')]]).float().to(device)

# sample = pc_sampler(score_model, sde, device, shape=(num_samples, 1, 28, 28))
noisy_samples, samples = pCN_sample(pc_sampler, num_mcmc_steps, burn_in_period, score_model, sde, 
                                    device, start_sample, y_use, H_use, std_y, beta=0.25)

# np.savetxt('bin/noisy_sample.txt', noisy_samples.cpu().numpy())

nmse = compute_NMSE(y.to('cpu'), H_use.to('cpu'), samples.to('cpu')).to('cpu')

sample_grid = make_grid(samples, nrow=int(np.sqrt(num_mcmc_steps - burn_in_period)))

plt.figure(figsize=(6,6))
plt.title('NMSE: {}'.format(nmse))
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.savefig("sample.png")
