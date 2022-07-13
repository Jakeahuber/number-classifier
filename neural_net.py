import torch.nn as nn

# Define the neural network
class Neural_Net(nn.Module):
    # Constructor
    def __init__(self):
        # Allows access to methods within nn.Module
        super(Neural_Net, self).__init__()

        # 1 input image channel, 10 output channels, 5x5 square convolution kernel
        # Not too sure what convolution kernel does. But standard is either 5x5 or 3x3
        self.conv1 = nn.Conv2d(1, 10, 5)

        

    def forward_prop(self, x):
        print("TODO")