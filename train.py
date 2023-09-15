from matplotlib import pyplot as plt
import torch
from network import ScoreNet
from sampling import *
from torchvision.utils import *
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as tvu

from argparse import ArgumentParser
import os
import yaml
from sde import VPSDE
from utils import dict2namespace


def loss_fn(model, x, sde, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x).to(x.device)
  std = sde.marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


def train(model, sde, trainloader, device, config):

    print("Training starting...")

    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    ckpt_dir = os.path.join(basedir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    fig_dir = os.path.join(basedir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)

    optimizer = Adam(model.parameters(), lr=config.optim.lr)

    pbar = tqdm(range(config.train.num_epoch))
    for i in pbar:

        total_loss = 0.0

        for data, y in trainloader:

            data = data.to(device)
            optimizer.zero_grad()

            loss = loss_fn(model, data, sde, eps=config.sampling.eps)
            loss.backward()

            if config.optim.grad_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.optim.grad_clip)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        pbar.set_description(
            (
                f'Epoch: {i}. Train loss: {avg_loss}'
            )
        )

        if i % config.train.save_step == 0 and i > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{i}.pt')
            torch.save(model.state_dict(), ckpt_path)

        if i % config.train.eval_step == 0 and i > 0:

            fig_path = os.path.join(fig_dir, f'{i}.png')

            samples = pc_sampler(model, sde, device, shape=(config.sampling.num_samples, 1, 28, 28), 
                                 probability_flow=config.sampling.probability_flow)
            sample_grid = make_grid(samples, nrow=int(np.sqrt(config.sampling.num_samples)))
            plt.figure()
            plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            plt.savefig(fig_path)
            plt.close()

    print("Training done")

    if config.model.ckpt != "none":
        torch.save(model.state_dict(), config.model.ckpt)
    else:
        torch.save(model.state_dict(), 'model.pt')


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    sde = VPSDE(config.model.num_scales, beta_min=config.model.beta_min, beta_max=config.model.beta_max)

    if device.type == "cpu":
        model = ScoreNet(sde=sde, device=device)
    else:
        model = torch.nn.DataParallel(ScoreNet(sde=sde, device=device), device_ids=[0])

    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)

    data_loader = DataLoader(dataset, batch_size=config.train.batchsize, shuffle=True, num_workers=4) 

    if config.model.ckpt != 'none':
          model.load_state_dict(torch.load(config.model.ckpt))

    train(model, sde, data_loader, device, config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian.yml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='none')
    args = parser.parse_args()
    subprocess(args)