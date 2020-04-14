import scipy.io as sio
import numpy as np
import torch




def load_data(filename, batch_size):
    mat_content = sio.loadmat(filename)
    train_data = mat_content['train_data']
    train_data = train_data[0]
    train_label = mat_content['train_label']
    train_label = train_label[0]
    test_data = mat_content['test_data']
    test_data = test_data[0]
    test_label = mat_content['test_label']
    test_label = test_label[0]


    num_train = train_data.shape[0]
    num_test = test_data.shape[0]
    input_size = train_data[1].shape[1]
    print(input_size)

    train_itr = []

    for i in range(0,num_train, batch_size):
        if (i+batch_size)>num_train:
            current_data = train_data[i:num_train]
            current_label = train_label[i:num_train]
        else:
            current_data = train_data[i:i+batch_size]
            current_label = train_label[i:i+batch_size]
        data_length =[len(sample) for sample in current_data]
        pad_data = np.zeros((len(data_length), max(data_length), input_size))
        for i , sample_len in enumerate(data_length):
            sample = current_data[i]

            pad_data[i, 0:sample_len, :] = sample

        data = torch.from_numpy(pad_data).float()
        print('data: ', data.shape)
        label = torch.from_numpy(current_label)
        print('label: ', label.shape)
        current_batch = [data, label]
        train_itr.append(current_batch)


    test_itr = []

    for i in range(0,num_test, batch_size):
        if (i+batch_size)>num_test:
            current_data = test_data[i:num_test]
            current_label = test_label[i:num_test]
        else:
            current_data = test_data[i:i+batch_size]
            current_label = test_label[i:i+batch_size]
        data_length = [len(sample) for sample in current_data]
        pad_data = np.zeros((len(data_length), max(data_length), input_size))
        for i, sample_len in enumerate(data_length):
            sample = current_data[i]

            pad_data[i,0:sample_len, :] = sample

        data = torch.from_numpy(pad_data).float()
        print('data: ', data.shape)
        label = torch.from_numpy(current_label)
        print('label: ', label.shape)
        current_batch = [data, label]
        test_itr.append(current_batch)

    return train_itr, test_itr