import torch
from torch.utils.data import Dataset
import utils
import os
# import ipdb
import glob
import time
import random
import sys
from PIL import Image
import numpy as np
print(sys.path.pop(4))
import cv2

import glob
# load image and convert to and from NumPy array
from numpy import asarray
# End Imports
# ----------------------------------------------------

# File Global Variables and datapath setting
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path,"Data")

data_path_train = os.path.join(dir_path, "Data", "fruits-360_dataset", "fruits-360", "Training")
data_path_test = os.path.join(dir_path, "Data", "fruits-360_dataset", "fruits-360", "Test")
sys.path.insert(0, data_path)


# =================================================================================================================
# Class Declaration
# =================================================================================================================
class Fruits(Dataset):
    
    def __init__(self, inputs= None, mode = 'train', dataDir=None, csv_file=None, valPerCent =0.1, normalize= True):
        
        if inputs:
            if isinstance(inputs, list):
                if len(inputs) != 4:
                    print("Error! Dataset inputs should be a list of len 4 with the format: train data, target, mean, std \n")
                self.data   = torch.from_numpy(inputs[0]).float()
                self.target = torch.from_numpy(inputs[1]).long().flatten()
                self.mean = inputs[2]
                self.std  = inputs[3]
        else:
            #returns are: train_x, train_y, test_x, test_y, train_mean, train_std, test_mean, test_std
            returns = load_dataset(data_path_train, data_path_test)
        # Argument handling
            if mode == 'train': 
                self.data   = torch.from_numpy(inputs[0]).float()
                self.target = torch.from_numpy(inputs[1]).long()
                self.mean = inputs[2]
                self.std  = inputs[3]
            else:
                self.data   = torch.from_numpy(inputs[4]).float()
                self.target = torch.from_numpy(inputs[5]).long()
                self.mean = inputs[6]
                self.std  = inputs[7]
        print(len(inputs))
        print(self.data.shape)
        self.mode = mode
        self.N = self.data.shape[0]
        self.data=self.data.permute(0,3,1,2)
        if normalize:
            self.data /= 255
        #self.rand_idx = [ i for i in range(0, self.num_data)]
        #random.Random(123).shuffle(self.rand_idx)
        print("Data Loaded: {}. Data instance shape is: {}\n".format( self.N, self.data[0].shape))
    # --------------------------------------------------------------------------------
    def __read_image_data(self):
        """ Not currently used!
        """
        
        trainImgs = [[] for i in range(len(self.dataFolders.keys()))]
        retval = os.getcwd()
        labels = {'mapping':[], 'itemLabels':[]}
        
        for i, fruitCat in enumerate(self.dataFolders):
            if i == 2:
                break
            labels['mapping'].append((fruitCat,i))
            folder = os.path.join(self.dataDir, fruitCat)
            print(folder)
            # Now change the directory
            os.chdir( folder )
            trainImgs[i].append([np.asarray(cv2.imread(img)) for img in os.listdir(folder)])
            
        trainImgs = np.asarray(trainImgs)
        os.chdir(retval)
        print(labels)
        return trainImgs, labels 
    
    # --------------------------------------------------------------------------------

   
    # --------------------------------------------------------------------------------
    def _transform_data_to_tensor(self):
        temp = torch.ones_like(self.data[0][0])
    # --------------------------------------------------------------------------------
    def __getitem__(self, index):
        """ DESCRIPTION: Mandatory function for a PyTorch dataset to implement. This is called by the dataloader iterator to return
                         batches of data for training.
        """
        data   = self.data[index]
        target = self.target[index]
        # In this framework dataloaders should return: data, target, misc
        return [data, target]

    # --------------------------------------------------------------------------------

    def __len__(self):
        return self.N
# End of Fruits class
# ====================================================================================================================

def _load_data(path, label_map=None):
    """
    :param path: sees all the folders in 'path' as labels to be assigned to images in respective folders
    directory structure should look like this: path/[label]/[images_for_that_label]
    :param label_map:  label map (optional) tells which number to be assigned to which numeric labels
    must be a dict. label_map[name_of_label] = unique_number_for_label
    :return: data_array, numeric_labels_array, label_map
    use of label map only is useful in case there are labels not present in testing data, for example.
    e.g.
    > train_x, train_y, label_map = load_data('./dataset/train')
    now use the same map for data
    > test_x, test_y, _ = load_data('./dataset/test', label_map)
    """

    labels = os.listdir(path)
    labels.sort()

    # pre-empt the size for faster arrays.
    all_files = glob.glob(path + '/*/*.jpg')
    n = len(all_files)

    images = np.zeros((n, 100, 100, 3), dtype=np.uint8)
    img_labels = np.zeros(n, dtype=np.uint8)

    # data_labels = []
    if label_map:
        user_map = True
    else:
        user_map = False
        label_map = {}

    label_i = 0
    img_i = 0
    sum_arr = np.zeros((100,100,3), dtype=np.float64)
    sq_arr = np.zeros((100,100,3), dtype=np.float64)
    for label in labels:
        if label[0] == '.':
            # ignore folders beginning with 'dot' e.g. '.DS_Store' in Mac OS
            continue

        if not user_map:
            label_map[label] = label_i
            label_i += 1
        img_paths = os.listdir(path + '/' + label)

        for img_fname in img_paths:
            img_arr = np.array(Image.open('{}/{}/{}'.format(path, label, img_fname)))
            # img_arr = imread('{}/{}/{}'.format(path, label, img_fname))
            # volumes.append(img_arr)
            images[img_i, :, :, :] = img_arr
            float_arr = img_arr / 255
            sum_arr += float_arr
            sq_arr += float_arr ** 2
            img_labels[img_i] = label_map[label]
            img_i += 1
            # img_labels.append(label_map[label])

    mean = sum_arr / img_i
    mean_sq = sq_arr / img_i
    var = mean_sq - (mean ** 2)
    std = np.sqrt(var)
    return images, img_labels, label_map, mean, std

# ---------------------------------------------------------------------------------------
def load_dataset(data_package_path = None, train_path = None, test_path = None ):
    
    data_package_path = data_package_path if data_package_path is not None else os.path.join(data_path,'dataset.npz')
    # Check if packaged data exists, else make it. Big time saver!
    print("Looking for packaged data at {}".format(data_package_path))
    if (data_package_path is None) or (not os.path.exists(data_package_path)):
        print(os.path.join(data_path,'dataset.npz'))
        print(os.path.exists(os.path.join(data_path,'set.npz')))
        if not os.path.exists(data_package_path):
            print("Reading all files. This may take a couple of minutes. Please wait.")
            train_path = train_path if train_path is not None else data_path_train
            test_path  = test_path if test_path is not None else data_path_test
            t0 = time.time()
            train_x, train_y, label_map, train_mean, train_std = _load_data(train_path)
            t1 = time.time()
            print("{:.2f} loaded training data".format(t1 - t0))
            test_x, test_y, _, test_mean, test_std = _load_data(test_path, label_map)
            t2 = time.time()
            print("{:.2f} loaded testing data".format(t2 - t1))
            # train_mean = np.mean(train_x, axis=0)
            # train_std = np.std(train_x, axis=0)
            #
            # test_mean = np.mean(test_x, axis=0)
            # test_std = np.mean(test_x, axis=0)
            t3 = time.time()
            print("{:.2f} calculated statistics for data".format(t3 - t2))
            np.savez('dataset.npz',
                     train_x=train_x,
                     train_y=train_y,
                     test_x=test_x,
                     test_y=test_y,
                     train_mean=train_mean,
                     train_std=train_std,
                     test_mean=test_mean,
                     test_std=test_std)
            t4 = time.time()
            print("{:.2f} saved as a single file 'dataset.npz' for quick access".format(t4 - t3))
    else: # if data package exists load it!
        print("Existing file 'dataset.npz' found.")
        t0 = time.time()
        ds = np.load(data_package_path)
        train_x = ds['train_x']
        train_y = ds['train_y']
        test_x = ds['test_x']
        test_y = ds['test_y']
        train_mean = ds['train_mean']
        train_std = ds['train_std']
        test_mean = ds['test_mean']
        test_std = ds['test_std']
        t1 = time.time()
        print("{:.2f} Loaded data.".format(t1 - t0))

    return train_x, train_y, test_x, test_y, train_mean, train_std, test_mean, test_std


#=================================================================
# Main
# ================================================================
def main():
    
    data = load_dataset()
    dSet = Fruits(inputs=list(data[0:4]))
    
    print(dSet, dSet.data.shape)
    print(dSet.__getitem__(5)[1])
    # --------------------

if __name__ == "__main__":
    main()
