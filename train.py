from matplotlib import pyplot as plt
import torch
from torchvision import transforms

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


def train(model, sde, trainloader, optimizer, device, num_epochs, eps=1e-5,
          grad_clip=1., file_in="none", file_out="none"):
    if file_in != "none":
        model.load_state_dict(torch.load(file_in))
        print("loading model...")
    else:
        print("not loading model")

    print("Training starting...")

    for i in range(1, num_epochs):

        total_loss = 0.0

        for data, y in trainloader:

            data = data.to(device)
            optimizer.zero_grad()

            # print(data.shape)

            loss = loss_fn(model, data, sde, eps=eps)
            loss.backward()

            if grad_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        print("Iteration {} - Average error: {}".format(i, avg_loss))

    print("Training done")

    if file_out != "none":
        torch.save(model.state_dict(), file_out)
    else:
        torch.save(model.state_dict(), 'checkpoints/model.pth')
