import torch
from neural_net import Net, HDF5DataSet

batch_size = 1
neural_net = Net()
neural_net.load_state_dict(torch.load('model.pth'))

testing_set = HDF5DataSet('data-set/testset.hdf5')
testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=False)

correct = 0
total = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testing_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = neural_net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 200 test images: {100 * correct // total} %')