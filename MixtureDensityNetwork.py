import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


class MDN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_gaussians):
        super(MDN, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.n_gaussians = n_gaussians
        self.n_output = n_output
        # IMPORTANT notes
        # - Use softmax activation for pi  (they need to add up to 1)
        # - Use exponential linear unit for deviations (they should not be negative or close to zero)
        self.pis = nn.Linear(n_hidden, n_gaussians)  # Coefficents
        self.mus = nn.Linear(n_hidden, n_gaussians * n_output)  # Means
        self.sigmas = nn.Linear(n_hidden, n_gaussians)  # Variances / Deviations

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        pi = func.softmax(self.pis(x), -1)
        sigma = torch.exp(self.sigmas(x))
        mu = self.mus(x)

        return pi, sigma, mu

    def predict(self, inputs):
        pi, sigma, mu = self.forward(inputs)
        return self.sample_preds(pi.data.numpy(), sigma.data.numpy(), mu.data.numpy())

    @staticmethod
    def gaussian_pdf(x, mu, sigma):
        return (1 / torch.sqrt(2 * np.pi * sigma)) * torch.exp((-1 / (2 * sigma)) * torch.norm((x - mu), 2, 1) ** 2)

    def sample_preds(self, pi, sigma, mu):
        pred = np.zeros(self.n_output)
        pi_index = np.random.choice(range(len(pi)), p=pi)
        for t in range(self.n_output):
            pred[t] = np.random.normal(mu[pi_index * t:(pi_index + 1) * (t + 1)][0], sigma.data[pi_index])
        return pred
