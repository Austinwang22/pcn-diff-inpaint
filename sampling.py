import torch
import numpy as np
from scipy import integrate


def pc_sampler(score_model, sde, device, shape=(1, 1, 28, 28), snr=0.16, n_steps=1, 
               eps=1e-3, x_start=None, probability_flow=True):

    with torch.no_grad():

        if x_start is None:
            x = torch.randn(shape).to(device) * \
                sde.marginal_prob_std(torch.ones(shape[0], device=device))[:, None, None, None]
        else:
            x = x_start

        time_steps = torch.linspace(sde.T(), eps, sde.N, device=device)

        for i in range(sde.N):
            t = time_steps[i]
            vec_t = torch.ones(shape[0], device=device) * t

            if probability_flow:
                x, x_mean = euler_ode_update(x, vec_t, score_model, sde)
            else:
                x, x_mean = euler_maruyama_update(x, vec_t, score_model, sde)

        return x_mean


def euler_maruyama_update(x, t, score_model, sde):
    dt = -1. / sde.N
    z = torch.randn_like(x)
    drift, diffusion = sde.sde(x, t)
    score = sde.score_fn(score_model, x, t)
    drift = drift - diffusion[:, None, None, None] ** 2 * score

    x_mean = x - (diffusion**2)[:, None, None, None] * score * dt
    x = x_mean + torch.sqrt(torch.tensor(-dt)) * diffusion[:, None, None, None] * z

    return x, x_mean


def euler_ode_update(x, t, score_model, sde):
    dt = -1. / sde.N
    drift, diffusion = sde.sde(x, t)
    score = sde.score_fn(score_model, x, t)
    drift = drift - 0.5 * diffusion[:, None, None, None] ** 2 * score
    x_mean = x + drift * dt
    x = x_mean
    return x, x_mean


def ode_sampler(score_model, sde, device, shape=(64,2), atol=1e-5, rtol=1e-5, eps=1e-3, x_start=None):

    print("ODE Sampling...")

    if x_start is None:
        x = torch.randn(shape).to(device)
        print("\tRandomly generating initial x")
    else:
        x = x_start
        print("\tUsing preset initial x")

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device,
                                  dtype=torch.float).reshape((sample.shape[0],))
        with torch.no_grad():
            # score = sde.score_fn(score_model, sample, time_steps)
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def sde_eval_wrapper(sample, time_steps):
        sample = torch.tensor(sample, device=device, dtype=torch.float).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device,
                                  dtype=torch.float).reshape((sample.shape[0],))
        drift, diffusion = sde.sde(sample, time_steps)
        return drift.cpu().numpy().reshape((-1,)).astype(np.float64), \
            diffusion.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = torch.ones(shape[0], device=device, dtype=torch.float) * t
        score = score_eval_wrapper(x, time_steps)
        drift, diffusion = sde_eval_wrapper(x, time_steps)
        drift = drift - 0.5 * diffusion[0] ** 2 * score
        return drift

    res = integrate.solve_ivp(ode_func, (1., eps), x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol,
                              method='RK45')

    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x
