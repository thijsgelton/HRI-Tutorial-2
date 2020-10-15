import torch
import torch.nn as nn


class FFN(nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

    def predict(self, x):
        roll, pitch = self.forward(x)
        return roll.item(), pitch.item()
