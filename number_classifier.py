import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mnist import MNIST
from neural_net import Neural_Net
    

# Reads training set
mndata = MNIST('training-set')

'''
Each image has size 28x28 pixels (Area = 784). Images is a list of lists. Labels is a list of integer values 0-9. 
EX: images[0] contains 784 pixels/values.
EX: Labels[0] is the correct value of images[0].
'''
images, labels = mndata.load_training()

# Create tensors (similiar to numpy arrays)
X = torch.tensor(images) # Shape: [60,000, 784]
Y = torch.tensor(labels) # Shape: [60,000]

# Move tensors to the GPU if available
if torch.cuda.is_available():
    X = X.to('cuda')
    Y = Y.to('cuda')
    print(f"X tensor is stored on: {X.device}")
    print(f"Y tensor is stored on: {Y.device}")

model = torchvision.models.resnet(pretrained=True)
predictions = model(X) # Forward pass. Computes estimated Y values
loss = (predictions - labels).sum() # Computes loss
loss.backward() # Backward Pass. Computes derivatives for gradient descent. 

# Load in optimizer with a learning rate of 0.01 and momentum of 0.9.
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer.step() # Performs gradient descent



