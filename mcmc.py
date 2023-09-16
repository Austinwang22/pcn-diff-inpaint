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


def acceptance_prob(curr_sample, candidate, y, H, std_y):
    m1 = torch.norm(y - torch.matmul(H, curr_sample))
    m2 = torch.norm(y - torch.matmul(H, candidate))

    exp = torch.exp((m1 - m2) / (2. * std_y * std_y))
    ones = torch.ones_like(exp)
    accept = torch.minimum(exp, ones)

    return accept


def compute_NMSE(y, H, samples, eps=1e-6):
    Hx = torch.matmul(H, samples)
    nmse = torch.mean(torch.pow(Hx - y, 2)) / (torch.mean(torch.pow(y, 2))) + eps
    return nmse


def generate_candidate(curr_sample, beta):
    noise = torch.randn_like(curr_sample)
    return torch.sqrt(torch.tensor(1. - beta * beta)) * curr_sample + beta * noise


def pCN_sample(device, model, sde, start_x, config):

    noisy_posterior = []
    posterior = []

    y = torch.load(config.data.y_path).to(device)
    H = torch.load(config.data.H_path).to(device)
    std_y = config.data.std_y
    beta = config.sampling.beta

    curr_sample = start_x
    curr_denoised = pc_sampler(model, sde, device, shape=(1, 1, 28, 28), 
                               x_start=curr_sample, probability_flow=True)
    
    pbar = tqdm(range(config.sampling.N))
    for i in pbar:

        candidate = generate_candidate(curr_sample, beta).to(device)
        candidate_denoised = pc_sampler(model, sde, device, shape=(1, 1, 28, 28),
                                        x_start=candidate, probability_flow=True)
        
        accept = acceptance_prob(curr_denoised, candidate_denoised, y, H, std_y)
        rand = torch.rand_like(accept).to(device)

        pbar.set_description(
            (
                f'Iteration: {i}. Acceptance probability: {accept}'
            )
        )

        if rand < accept:
            curr_sample = candidate
            curr_denoised = candidate_denoised

            if i > config.sampling.burn_in:
                noisy_posterior.append(curr_sample)
                posterior.append(curr_denoised)

        if i == config.sampling.burn_in:
            noisy_posterior.append(curr_sample)
            posterior.append(curr_denoised)

        if i > 0 and i % config.sampling.save_step == 0:
            save_path = os.path.join(config.sampling.save_path, f'{i}.png')
            tvu.save_image(curr_denoised, save_path)

    noisy_posterior = torch.cat(noisy_posterior)
    posterior = torch.cat(posterior)
    nmse = compute_NMSE(y, H, posterior)
    return noisy_posterior, posterior, nmse


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    ckpt_dir = os.path.join(basedir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    fig_dir = os.path.join(basedir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)

    sde = VPSDE(config.model.num_scales, beta_min=config.model.beta_min, beta_max=config.model.beta_max)

    if device.type == "cpu":
        model = ScoreNet(sde=sde, device=device)
    else:
        model = torch.nn.DataParallel(ScoreNet(sde=sde, device=device), device_ids=[0])

    print("Loading preset weights from {}".format(config.model.ckpt))
    model.load_state_dict(torch.load(config.model.ckpt))

    start_x = torch.randn((1, 1, 28, 28), device=device)

    noisy_posterior, posterior, nmse = pCN_sample(device, model, sde, start_x, config)

    np_fig_path = os.path.join(fig_dir, 'noisy_posterior.png')
    p_fig_path = os.path.join(fig_dir, 'posterior.png')
    np_ckpt_path = os.path.join(ckpt_dir, 'noisy_posterior.pt')
    p_ckpt_path = os.path.join(ckpt_dir, 'posterior_ckpt.pt')

    tvu.save_image(noisy_posterior, np_fig_path)
    torch.save(noisy_posterior, np_ckpt_path)
    torch.save(posterior, p_ckpt_path)

    plt.figure()
    plt.title('Sampled Images\nNMSE: {}'.format(nmse))
    sample_grid = make_grid(posterior, nrow=int(np.sqrt(posterior.shape[0])))
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.savefig(p_fig_path)
    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian.yml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='none')
    args = parser.parse_args()
    subprocess(args)