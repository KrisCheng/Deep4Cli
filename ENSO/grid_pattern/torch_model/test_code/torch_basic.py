# Author: Kris Peng
# Created on 24/12/2018
# ref:https://github.com/yunjey/pytorch-tutorial

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# # 1. autograd
# x = torch.tensor(2., requires_grad=True)
# w = torch.tensor(5., requires_grad=True)
# b = torch.tensor(3., requires_grad=True)
#
# y = w * x + b
#
# y.backward()
#
# print(x.grad)
# print(w.grad)
# print(b.grad)
#
# x = torch.randn(10, 4)
# y = torch.randn(10, 2)
# linear = nn.Linear(4, 2)
# print('w: ', linear.weight)
# print('b: ', linear.bias)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
#
# pred = linear(x)
#
# loss = criterion(pred, y)
# # print('y: ', y)
# # print('pred: ', pred)
# print('loss: ', loss.item())
#
# loss.backward()
# # print(linear)
# print('dL/dw: ', linear.weight.grad)
# print('dL/db: ', linear.bias.grad)
#
# optimizer.step()
#
# pred = linear(x)
# loss = criterion(pred, y)
# print('loss after 1 step optimization: ', loss.item())

# # 2. numpy --> PyTorch
# x = np.array([[1, 2], [3, 4]])
# y = torch.from_numpy(x)
# z = y.numpy()
# print(x)

# # 3. linear regression
# # Hyper-parameters
# input_size = 1
# output_size = 1
# num_epochs = 60000
# learning_rate = 0.001
#
# # Toy dataset
# x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
#                     [9.779], [6.182], [7.59], [2.167], [7.042],
#                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
#
# y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
#                     [3.366], [2.596], [2.53], [1.221], [2.827],
#                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
#
# # Linear regression model
# model = nn.Linear(input_size, output_size)
#
# # Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # Train the model
# for epoch in range(num_epochs):
#     # Convert numpy arrays to torch tensors
#     inputs = torch.from_numpy(x_train)
#     targets = torch.from_numpy(y_train)
#
#     # Forward pass
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#
#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch+1) % 5 == 0:
#         print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
#
# # Plot the graph
# predicted = model(torch.from_numpy(x_train)).detach().numpy()
# plt.plot(x_train, y_train, 'ro', label='Original data')
# plt.plot(x_train, predicted, label='Fitted line')
# plt.legend()
# plt.show()
#
# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')

# # 4. logistic regression
# # Hyper-parameters
# input_size = 784
# num_classes = 10
# num_epochs = 1
# batch_size = 100
# learning_rate = 0.001
#
# # MNIST dataset (images and labels)
# train_dataset = torchvision.datasets.MNIST(root='data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='data',
#                                           train=False,
#                                           transform=transforms.ToTensor())
#
# # Data loader (input pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
#
# # Logistic regression model
# model = nn.Linear(input_size, num_classes)
#
# # Loss and optimizer
# # nn.CrossEntropyLoss() computes softmax internally
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# # Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # Reshape images to (batch_size, input_size)
#         images = images.reshape(-1, 28*28)
#
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i+1) % 100 == 0:
#             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
#
# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()
#
#     print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
#
# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')

# 5. FNN Model
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 600
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
model = nn.DataParallel(model)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
