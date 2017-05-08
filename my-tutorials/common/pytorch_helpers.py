import torch.nn as nn

DATA_PATH = '../tutorials/data/'


class LinearRegression(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.linear(x)

    def __call__(self, *args, **kwargs):
        return super(LinearRegression, self).__call__(*args, **kwargs)


# Neural Network Model (1 hidden layer)
class Net1(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out