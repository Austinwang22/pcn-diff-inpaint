from matplotlib import pyplot as plt
import torch
import numpy as np
import torchvision.utils as tvu


def forward_operator(H, x):
    return torch.matmul(H, x)


def compute_NMSE(y, H, samples, eps=1e-6):
    Hx = forward_operator(H, samples)
    nmse = torch.mean(torch.pow(Hx - y, 2)) / (torch.mean(torch.pow(y, 2))) + eps
    return nmse


def acceptance_prob(curr_sample, candidate, y, H, std_y):
    m1 = torch.norm(y - torch.matmul(H, curr_sample))
    m2 = torch.norm(y - torch.matmul(H, candidate))

    exp = torch.exp((m1 - m2) / (2. * std_y * std_y))
    ones = torch.ones_like(exp)
    accept = torch.minimum(exp, ones)

    print("Mean acceptance prob: {}".format(torch.mean(accept)))
    return accept


def generate_candidate(curr_sample, beta=0.5):
    noise = torch.randn_like(curr_sample)
    return torch.sqrt(torch.tensor(1 - beta * beta)) * curr_sample + beta * noise


def pCN_sample(sample_fn, N, burn_in_period, score_model, sde, device, start_sample, y, H, std_y, beta=0.5):

    curr_sample = start_sample.to(device)
    curr_denoised = sample_fn(score_model, sde, device, x_start=curr_sample)[0][0]

    approx_samples = []
    approx_samples_noise = []

    for i in range(N):

        candidate = generate_candidate(curr_sample, beta=beta).to(start_sample.device)
        candidate_denoised = sample_fn(score_model, sde, device, x_start=candidate)[0][0]

        tvu.save_image(candidate_denoised, f'bin/candidate{i}.png')

        accept = acceptance_prob(curr_denoised, candidate_denoised, y, H, std_y)

        # if i < burn_in_period:
        #     rand = torch.rand_like(accept).to(start_sample.device)
        #     mask1 = (accept <= rand).float().to(start_sample.device)
        #     mask2 = (accept > rand).float().to(start_sample.device)
        # else:
        #     mask1 = 0.
        #     mask2 = 1.
        rand = torch.rand_like(accept).to(start_sample.device)
        mask1 = (accept <= rand).float().to(start_sample.device)
        mask2 = (accept > rand).float().to(start_sample.device)

        curr_sample = curr_sample * mask1 + candidate * mask2
        curr_denoised = sample_fn(score_model, sde, device, x_start=curr_sample)[0][0]

        tvu.save_image(curr_denoised, f'bin/curr_sample{i}.png')

        if i >= burn_in_period:
            approx_samples.append(curr_denoised.tolist())
            approx_samples_noise.append(curr_sample.tolist())
            print('Using this sample')

            filename = f'bin/samples2/sample{i - burn_in_period}.png'
            tvu.save_image(curr_denoised, filename)

    # return torch.tensor(approx_samples_noise)[0][0][0], torch.tensor(approx_samples)
    return curr_sample[0][0], curr_denoised