"""
logistic regression (or logit) is a regression where output is categorical
"""
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from common.pytorch_helpers import DATA_PATH, LinearRegression

# Hyper Parameters
input_size = 784 # 28 * 28
num_classes = 10
num_epochs = 5
batch_size = 100

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root=DATA_PATH,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root=DATA_PATH,
                           train=False,
                           transform=transforms.ToTensor(),
                           download=False)

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = LinearRegression(input_size, num_classes)

training = not True
model_state_dict_filename = "models/mnist_logistic-params.pkl"

if training:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

            # forward, backward, optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                       % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # Save model
    torch.save(model.state_dict(), model_state_dict_filename)
else:
    model.load_state_dict(torch.load(model_state_dict_filename))

# Test model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28)) # size [100, 784]
    outputs = model(images) # size [100, 10]
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))