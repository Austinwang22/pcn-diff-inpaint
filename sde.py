import abc
import torch
import numpy as np


class SDE(abc.ABC):

  def __init__(self, N):
    '''
    Initializes the SDE.
    Args:
    N: the total number of time steps
    '''
    super().__init__()
    self.N = N

  @abc.abstractmethod
  def T(self):
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    '''Computes the drift and diffusion coefficients of dx = f(x,t)dt + g(t)dw'''
    pass

  @abc.abstractmethod
  def score_fn(self, model, x, t):
    '''Computes the score'''
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    '''Parameters to determine the marginal distribution of the SDE, $p_0t(x)$.'''
    pass

  @abc.abstractmethod
  def marginal_prob_std(self, t):
    pass


class VESDE(SDE):

  def __init__(self, N, sigma_min=0.02, sigma_max=100):
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

  def T(self):
    return 1

  def sde(self, x, t):
    '''
    Returns the drift and diffusion coefficients of the SDE at time t.
    '''
    f = torch.zeros(x.shape).to(x.device)

    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    g = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min))))

    return f, g

  def score_fn(self, model, x, t):
    score = model(x, t)
    std = self.marginal_prob_std(t)
    # return -score / std[:, None, None, None]
    # return score / std[:, None, None, None]
    return score

  def marginal_prob(self, x, t):
    '''
    A function that takes time t and gives the mean and std 
    of the perturbation kernel p_{0t}(x(t) | x(0)).
    '''
    mean = x

    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    return mean, std

  def marginal_prob_std(self, t):
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** t


class VPSDE(SDE):

  def __init__(self, N, beta_min=0.1, beta_max=20):
    super().__init__(N)
    self.beta_min = beta_min
    self.beta_max = beta_max
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  def score_fn(self, model, x, t):
    std = self.sqrt_1m_alphas_cumprod.to(x.device)[t.long()]
    score = model(x, t)
    # return -score / std[:, None, None, None]
    # return score / std[:, None, None, None]
    return score

  def T(self):
    return 1.

  def sde(self, x, t):
    beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * \
                     (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def marginal_prob_std(self, t):
    log_mean_coeff = -0.25 * t ** 2 * \
                     (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
    return torch.sqrt(1. - torch.exp(2. * log_mean_coeff))



