import torch
import torch.nn as nn
import torch.optim as optim
from neural_net import Neural_Net
from torchvision import datasets
from torchvision.transforms import ToTensor
    
num_epochs = 200
learning_rate = 1
momentum = 0.9
batch_size = 256

# Create datasets for training & validation
# There are 60,000 images in the training set. Each one is 28x28, or 784 pixels.
training_set = datasets.MNIST(
    root='training-set/',
    train=True,
    download=True,
    transform=ToTensor()
)

# Why do i shuffle the training loader?
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

# There are 10,000 items in the testing set.
testing_set = datasets.MNIST(
    root='training-set/',
    train=False,
    download=True,
    transform=ToTensor() 
)

testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=False)

# Class labels
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

neural_net = Neural_Net() # Initialize neural network
#neural_net.load_state_dict(torch.load('model.pth'))

criterion = nn.CrossEntropyLoss() # Initialize loss function. This uses a cross entropy loss
optimizer = optim.SGD(neural_net.parameters(), lr=learning_rate, momentum=momentum) # Initialize optimizer

for epoch in range(num_epochs): # loop over training set num_epoch times
    running_loss = 0.0
    for i, data in enumerate(training_loader, 0): 
        inputs, labels = data
        inputs = inputs.view(-1, 28*28).requires_grad_()

        optimizer.zero_grad()
        outputs = neural_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    # print statistics    
    print(f'epoch: {epoch + 1}, loss: {running_loss / (60000 / batch_size):.3f}')

PATH = './model.pth'
torch.save(neural_net.state_dict(), PATH)
print('Finished Training')







