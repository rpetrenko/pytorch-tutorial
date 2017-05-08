import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from common.pytorch_helpers import DATA_PATH, Net1


# Hyper Parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
model_state_dict_filename = 'models/mnist_feedforward-params.pkl'

# MNIST Dataset
train_dataset = dsets.MNIST(root=DATA_PATH,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root=DATA_PATH,
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


model = Net1(input_size, hidden_size, num_classes)
model.cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
training = True
if training:
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, 28*28))
            images = images.cuda()
            labels = Variable(labels)
            labels = labels.cuda()

            # forward bacward optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss =criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
    # Save the Model
    torch.save(model.state_dict(), model_state_dict_filename)
else:
    model.load_state_dict(torch.load(model_state_dict_filename))

# Test model
correct = total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    images = images.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


