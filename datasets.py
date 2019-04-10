import pickle
import random
import numpy as np 
import os.path

def create_simple_dataset(*args):
    random.seed(0)
 
    data_x = None
    data_y = []
    for x in range(100):
        for y in range(100):

            features = np.array([y/100, x/100])
            if data_x is not None:
                data_x = np.vstack((data_x, features))
            else:
                data_x = features.copy()

            if x > y:
                data_y.append(0)
            else:
                data_y.append(1)
 
    data_x = np.transpose(data_x)

    return data_x, data_y

def get_cifar_dataset(batch_name, *args):
    file_path = os.path.join('cifar-10', batch_name)

    with open(file_path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

    cifar_x = np.transpose(dict[b'data'])
    cifar_y = np.array(dict[b'labels'])

    cifar_x_mean0 = cifar_x - np.mean(cifar_x, axis=0)
    cifar_x_standardized = cifar_x_mean0 / np.std(cifar_x, axis=0)

    return cifar_x_standardized, cifar_y
