Experiments with score-based generative models and the MNIST dataset. Code partially based on [this tutorial](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=LZC7wrOvxLdL).

# Train 
```bash
python3 train.py --config configs/mnist.yml
```

Example (Samples from model):

![Alt text](images/40.png)

# Inverse Problems
```bash
python3 train.py --config configs/mcmc-mnist-half.yml
```

Example Inpainting Results:

![Alt text](images/inpainting.png)
