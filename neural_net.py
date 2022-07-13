import torch
import torch.nn as nn

# Define the neural network. Uses logistic regression model
class Neural_Net(nn.Module):
    # Constructor
    def __init__(self):
        # Allows access to methods within nn.Module
        super(Neural_Net, self).__init__()

        # 784 - size of each input sample (image of size 28x28)
        # 1 - size of each output sample (number 0-9)
        self.linear = nn.Linear(28, 1, bias=True)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        return x