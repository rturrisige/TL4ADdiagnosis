import numpy as np
from glob import glob as gg
import torch
import os
import sys
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage


# ########################
#  CUSTOMIZED FUNCTIONS ##
# ########################

def my_data_list(data_path, img_format='.npy'):
    """
    It takes the path to the dataset and returns two lists of files, corresponding to CN and AD groups.
    This function assumes that all CN sample files start with label 0 and AD samples starts with label 2.
    :param data_path: path to image files, type=str
    :param img_format: file image format, type=str
    :return: lists
    """
    CN = gg(data_path + '/0*' + img_format)
    AD = gg(data_path + '/2*' + img_format)
    if len(CN) == 0 or len(AD) == 0:
        print('Empty folder. Please choose another folder containing npy files to process.')
        sys.exit()
    return CN, AD


def my_augmented_data_finder(train_list, augmented_dir, img_format='.npy'):
    """
    It takes the list of training samples and look for the corresponding augmented images.
    :param train_list: list of strings corresponding to training sample files, type=list
    :param augmented_dir: path to augmented dataset, type=str
    :param img_format: format in which images are saved, type=str
    :return: list of strings corresponding to augmented files, type=list
    """
    augment_list = []
    for f in train_list:
        name = os.path.basename(f).split(img_format)[0]
        augment_list += gg(augmented_dir + '/' + name + '*' + img_format)
    return augment_list


# ######################
#  DATA PROCESSING    ##
# ######################


def resize_data_volume_by_scale(data, scale):
    """
    :param data: image, type=numpy array
    :param scale:  value between 0 and 1, type=float
    :return: processed image, type=numpy array
    """
    if isinstance(scale, float):
        scale_list = [scale, scale, scale]
    else:
        scale_list = scale
    return ndimage.interpolation.zoom(data, scale_list, order=0)


def normalize_intensity(img_tensor, normalization="mean"):
    """
   Accepts an image tensor and normalizes it
   :param img_tensor: type=torch tensor
   :param normalization: choices = "max", "mean" , type=str
   For mean normalization we use the non zero voxels only.
   """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val
    elif normalization == "max":
        MAX, MIN = img_tensor.max(), img_tensor.min()
        img_tensor = (img_tensor - MIN) / (MAX - MIN)
    return img_tensor


def img_processing(image, scaling=0.5, final_size=[96, 96, 73]):
    image = resize_data_volume_by_scale(image, scale=scaling)
    new_scaling = [final_size[i] / image.shape[i] for i in range(3)]
    final_image = resize_data_volume_by_scale(image, scale=new_scaling)
    return final_image


def torch_norm(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    input_tensor = normalize_intensity(input_tensor)
    return input_tensor.unsqueeze_(0)


# ######################
#     DATA LOADING    ##
# ######################


class MyDataLoader(Dataset):
    def __init__(self, dataset, nclasses=2, transform=None, preprocessing=False, nchannels=1):
        """
        dataset : list of all filenames
        transform (callable) : a function/transform that acts on the images (e.g., normalization).
        """
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.transform = transform
        self.num_classes = nclasses
        self.n_channels = nchannels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = np.load(self.dataset[index], allow_pickle=True)
        if len(batch) == 2:
            x_batch, y_batch = batch
        else:
            x_batch = batch
            y_batch = int(os.path.basename(self.dataset[index])[0])
        if self.preprocessing:
            x_batch = img_processing(x_batch)
        if self.transform:
            x_batch = self.transform(x_batch)
        if self.n_channels == 3:
            x_batch = torch.concat([x_batch, x_batch, x_batch], 0)
        if self.num_classes == 2:
            y_batch = np.where(y_batch == 2, 1, y_batch)
        return x_batch, y_batch