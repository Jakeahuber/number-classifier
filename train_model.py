import torch
import torch.nn as nn
import torch.optim as optim
from neural_net import HDF5DataSet, Net
import numpy as np    
import matplotlib.pyplot as plt

num_epochs = 6
learning_rate = 0.000005
momentum = 0.9
batch_size = 1

training_set = HDF5DataSet('data-set/trainset.hdf5')
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

# Class labels
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

neural_net = Net()

criterion = nn.CrossEntropyLoss() # Initialize loss function. This uses a cross entropy loss
optimizer = optim.SGD(neural_net.parameters(), lr=learning_rate, momentum=momentum) # Initialize optimizer

for epoch in range(num_epochs): # loop over training set num_epoch times
    loss_vector = []
    correct = 0
    total = 0

    for i, data in enumerate(training_loader, 0): 
        inputs, labels = data
        labels = labels.long()

        optimizer.zero_grad()
        outputs = neural_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_vector.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'epoch: {epoch + 1}, loss: {np.mean(loss_vector)}, accuracy: {100 * correct // total} %')

PATH = './model.pth'
torch.save(neural_net.state_dict(), PATH)
print('Finished Training')







