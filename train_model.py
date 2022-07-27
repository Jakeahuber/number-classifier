import torch
import torch.nn as nn
import torch.optim as optim
from neural_net import HDF5DataSet, Net
from test_model import test_model
import numpy as np    

num_epochs = 13
learning_rate = 0.000005
momentum = 0.9
batch_size = 1

training_set = HDF5DataSet('data-set/trainset.hdf5')
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

neural_net = Net()

criterion = nn.CrossEntropyLoss() # Initialize loss function. This uses a cross entropy loss
optimizer = optim.Adam(neural_net.parameters(), lr=learning_rate, weight_decay=0.0009) # Initialize optimizer. Weight decay adds L2 regularization to decrease overfitting

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

    print(f'Epoch: {epoch + 1}, loss: {np.mean(loss_vector)}, accuracy: {100 * correct // total} %')

    PATH = './model.pth'
    torch.save(neural_net.state_dict(), PATH)
    test_model()

print('Finished Training')







