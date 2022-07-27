import torch
from neural_net import Net, HDF5DataSet


def test_model():
    batch_size = 1
    neural_net = Net()
    neural_net.load_state_dict(torch.load('model.pth'))

    testing_set = HDF5DataSet('data-set/testset.hdf5')
    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            outputs = neural_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 200 test images: {100 * correct // total} %')

if __name__ == '__main__':
    test_model()