import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from common.pytorch_helpers import LinearRegression
# define a line with (0,0), (1,1)
def line45(x):
    return x


def randomize_point(x, y, dx_max, dy_max):
    dx = 2 * random.random() - 1
    dx *= dx_max
    dy = 2 * random.random() - 1
    dy *= dy_max
    return x + dx, y + dy


def generate_points(x_min, x_max, f, N):
    assert N > 1, 'number of points should be > 1'
    dx = (x_max - x_min) / float(N-1)
    x = x_min
    points = []
    while x <= x_max:
        y = f(x)
        points.append((x, y))
        x += dx
    return points


def randomize_points(points, dx_max, dy_max):
    res = []
    for x, y in points:
        res.append(randomize_point(x, y, dx_max, dy_max))
    return res


def split_x_y(points):
    x = []
    y = []
    for p in points:
        x.append(p[0])
        y.append(p[1])
    return x, y


def list_to_numpy(l, t=np.float32):
    return np.array([[x] for x in l], dtype=t)


points = generate_points(0, 1, line45, 100)
# print points
x_p, y_p = split_x_y(points)

points_rand = randomize_points(points, .2, .2)
# print points_rand
x_pr, y_pr = split_x_y(points_rand)


############ LINEAR REGRESSION #################
# let's create linear regression for the randomized points
x_train = list_to_numpy(x_p)
print("X shape" + str(x_train.shape)) # (100, 1)
y_train = list_to_numpy(y_pr)

# Linear (1 -> 1)
model = LinearRegression(x_train.shape[1], y_train.shape[1])

# define loss and optimizer
criterion = nn.MSELoss()
# stochastic gradient descent
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train
num_epochs = 300
inputs = Variable(torch.from_numpy(x_train))
targets = Variable(torch.from_numpy(y_train))
for epoch in range(num_epochs):
    # vars: epoch, outputs loss
    optimizer.zero_grad() # why do we have to clear grads from last step?
    outputs = model(inputs) # this call self.forward
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print('Epoch [%d/%d], Loss: %.5f' % (epoch+1, num_epochs, loss.data[0]))


predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.plot(x_train, y_p, label='Expected line')
plt.legend()
plt.show()
