import torch
import numpy as np
from scipy import integrate


def pc_sampler(score_model, sde, device, shape=(1, 1, 28, 28), snr=0.16, n_steps=1, eps=1e-3, x_start=None):

    print("Predictor corrector sampling...")

    with torch.no_grad():

        if x_start is None:
            print("\tGenerating random x_start")
            x = torch.randn(shape).to(device) * \
                sde.marginal_prob_std(torch.ones(shape[0], device=device))[:, None, None, None]
        else:
            print("\tUsing preset x_start")
            x = x_start

        time_steps = torch.linspace(sde.T(), eps, sde.N, device=device)

        for i in range(sde.N):
            t = time_steps[i]
            vec_t = torch.ones(shape[0], device=device) * t

            # x, x_mean = langevin_corrector_update(x, vec_t, score_model, sde, snr=snr, n_steps=n_steps)
            # x, x_mean = euler_maruyama_update(x, vec_t, score_model, sde)
            # x, x_mean = vpsde_ancestral_update(x, t, score_model, sde)
            x, x_mean = euler_ode_update(x, vec_t, score_model, sde)

        return x_mean


def Euler_Maruyama_sampler(score_model, sde, device, shape=(64, 1, 28, 28), eps=1e-3):
  
    t = torch.ones(shape[0], device=device)
    init_x = torch.randn(shape, device=device) \
    * sde.marginal_prob_std(t)[:, None, None, None]
  
    time_steps = torch.linspace(1., eps, sde.N, device=device)
    step_size = time_steps[0] - time_steps[1]

    x = init_x

    with torch.no_grad():

        for time_step in time_steps:      
      
            batch_time_step = torch.ones(shape[0], device=device) * time_step

            drift, diffusion = sde.sde(x, batch_time_step)
            score = score_model(x, batch_time_step)

            drift = drift - diffusion[:, None, None, None] ** 2 * score
            mean_x = x + (diffusion**2)[:, None, None, None] * score * step_size
            x = mean_x + torch.sqrt(step_size) * diffusion[:, None, None, None] * torch.randn_like(x)      
    return mean_x


def euler_maruyama_update(x, t, score_model, sde):
    dt = -1. / sde.N
    z = torch.randn_like(x)
    drift, diffusion = sde.sde(x, t)
    score = sde.score_fn(score_model, x, t)
    # score = score_model(x, t)
    drift = drift - diffusion[:, None, None, None] ** 2 * score

    # x_mean = x + drift * dt
    # x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
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


def ve_ddim(score_model, sde, device, shape=(64, 2), eta=1, x_start=None):

    print("VE DDIM sampling...")

    steps = torch.arange(sde.N)
    orig_t_steps = (sde.sigma_max ** 2) * ((sde.sigma_min ** 2 / sde.sigma_max ** 2) ** (steps / (sde.N - 1)))
    sigma_steps = torch.sqrt(orig_t_steps)
    sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
    t_steps = (torch.as_tensor(sigma_steps)) ** 2
    # t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    if x_start is None:
        x = torch.randn(shape).to(device)
    else:
        x = x_start

    for i in range(len(t_steps) - 1):

        t = t_steps[i].to(device)
        s = t_steps[i + 1].to(device)
        sigma_t = sigma_steps[i].to(device)
        sigma_s = sigma_steps[i + 1].to(device)
        x = ve_ddim_update(score_model, sde, x, s, t, sigma_s, sigma_t, eta, device)

    return x


def ve_ddim_update(score_model, sde, x_t, s, t, sigma_s, sigma_t, eta, device):
    vec_t = torch.ones(x_t.shape[0], device=device) * t
    z = torch.randn_like(x_t)

    # score = sde.score_fn(score_model, x_t, vec_t)
    score = score_model(x_t, vec_t)
    hat_x_t = x_t + sigma_t * sigma_t * score
    c_ts = torch.sqrt(sigma_s ** 2 * (sigma_t ** 2 - sigma_s ** 2) / (sigma_t ** 2))

    x_s = hat_x_t + eta * c_ts * z - sigma_t * torch.sqrt(sigma_s ** 2 - (eta ** 2) * (c_ts ** 2)) * score

    return x_s


def vpsde_ancestral_update(x, t, score_model, sde):

    timestep = (t * (sde.N - 1) / sde.T()).long()
    vec_t = torch.ones(x.shape[0], device=t.device) * t
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = sde.score_fn(score_model, x, vec_t)

    x_mean = (x + beta * score) / torch.sqrt(1. - beta)
    z = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta) * z

    return x, x_mean


def euler_maruyama(score_model, sde, shape=(64,2), eps=1e-3):

    print("Euler-Maruyama sampling...\n")

    with torch.no_grad():
        x = torch.randn(shape).to(score_model.device)
        time_steps = torch.linspace(sde.T(), eps, sde.N, device=score_model.device)
        dt = -1. / sde.N

        for i in range(sde.N):
            t = time_steps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            z = torch.randn_like(x)

            drift, diffusion = sde.sde(x, vec_t)
            score = sde.score_fn(score_model, x, vec_t)
            drift = drift - diffusion[:, None, None, None] ** 2 * score
            # drift = drift - diffusion ** 2 * score

            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
            # x = x_mean + diffusion * np.sqrt(-dt) * z

        return x_mean


def langevin_corrector_update(x, t, score_model, sde, is_vpsde=True, snr=0.16, n_steps=1):
    if is_vpsde:
        timestep = (t * (sde.N - 1)).long()
        alpha = sde.alphas.to(t.device)[timestep]
    else:
        alpha = torch.ones_like(t)

    for i in range(n_steps):
        grad = sde.score_fn(score_model, x, t)
        noise = torch.randn_like(x)
        grad_norm = torch.norm(grad.reshape(
            grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(
            noise.shape[0], -1), dim=-1).mean()
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean



