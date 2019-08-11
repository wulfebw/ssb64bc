"""This file contains dataset classes for the various types of data formats.

This includes:
1. Images.
2. Preprocessed images (as .pth files).
3. Preprocessed (recurrent) datasets as h5 files.
"""
import os

import cv2
import h5py
import numpy as np
import pandas as pd
import torch

import ssb64bc.datasets.utils as dataset_utils
import ssb64bc.formatting.utils as formatting_utils


class MultiframeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_filepath,
                 img_dir,
                 image_type="color",
                 transform=None,
                 action_transform=None,
                 debug_size=None):
        self.df = pd.read_csv(dataset_filepath)
        self.length = len(self.df) if debug_size is None else debug_size
        self.img_dir = img_dir
        self.transform = transform
        self.action_transform = action_transform
        self.image_encoding = dataset_utils.get_image_encoding(image_type)
        self.frame_keys = formatting_utils.get_frame_cols(self.df)
        self.action_keys = formatting_utils.get_act_cols(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_filenames = [filename for filename in self.df.loc[idx, self.frame_keys]]
        img_filepaths = [os.path.join(self.img_dir, filename) for filename in img_filenames]

        # Load and transform the images individually before stacking.
        images = [cv2.imread(img_filepath, self.image_encoding) for img_filepath in img_filepaths]
        if self.transform:
            # perform the transform to the images directly.
            images = [self.transform(image) for image in images]
        images = torch.cat(images, dim=0)

        # Load the actions.
        actions = self.df.loc[idx, self.action_keys]
        if self.action_transform is not None:
            actions = self.action_transform(actions)
        actions = torch.tensor(actions, dtype=torch.int64)
        return images, actions


class PreprocessedMultiframeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_filepath, tensor_dir, action_transform=None, debug_size=None):
        self.df = pd.read_csv(dataset_filepath)
        self.length = len(self.df) if debug_size is None else debug_size
        self.tensor_dir = tensor_dir
        self.action_transform = action_transform
        self.frame_keys = formatting_utils.get_frame_cols(self.df)
        self.action_keys = formatting_utils.get_act_cols(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        tensor_filenames = [filename for filename in self.df.loc[idx, self.frame_keys]]
        tensor_filepaths = [os.path.join(self.tensor_dir, filename) for filename in tensor_filenames]
        tensors = torch.cat([torch.load(f) for f in tensor_filepaths], dim=0)

        # Load the actions.
        actions = self.df.loc[idx, self.action_keys]
        if self.action_transform is not None:
            actions = self.action_transform(actions)
        actions = torch.tensor(actions, dtype=torch.int64)
        return tensors, actions


class HDF5PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, h5_filepath, debug_size=None):
        self.h5_filepath = h5_filepath
        h5file = h5py.File(h5_filepath, "r")
        self.length = len(h5file["imgs"]) if debug_size is None else debug_size
        self.transform = lambda x: torch.tensor(x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open the file each time to avoid multiprocessing issues.
        h5file = h5py.File(self.h5_filepath, "r")
        imgs = self.transform(h5file["imgs"][idx])
        action = h5file["actions"][idx]
        # Assumes the multiclass case by taking this argmax.
        action = np.argmax(np.array(action), axis=-1)
        action = torch.tensor(action, dtype=torch.int64)
        return imgs, action
