import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('dataset.hdf5', 'r') as f:
    image = np.array(f['0']['image'])
    print(np.array(f['0']['label']))
    print(type(image))

    plt.imshow(image)
    plt.show()

    image = np.array(f['1']['image'])
    print(np.array(f['1']['label']))
    print(type(image))

    plt.imshow(image)
    plt.show()

    image = np.array(f['2']['image'])
    print(np.array(f['2']['label']))
    print(type(image))

    plt.imshow(image)
    plt.show()