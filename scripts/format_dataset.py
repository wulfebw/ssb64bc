"""Format and optionally preprocess matches as n-frame datasets.

This script formats match into a dataset. This means that it pairs
actions with images (potentially multiple) and stores those pairs
in a dataframe. It also optionally preprocesses the dataset, which
means that it performs the e.g., torchvision transforms and saves
the images as .pth files.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import pathlib
import torch

from ssb64bc.datasets.utils import get_image_transforms
import ssb64bc.formatting.action_formatters as action_formatters
from ssb64bc.formatting.match_data import MatchData
from ssb64bc.formatting.multi_frame_dataset_formatter import MultiframeDatasetFormatter
from ssb64bc.formatting.utils import IMG_EXT, get_act_counts


def balance_noops_multiclass(df, max_fraction):
    """Balances the dataset by action counts in the multiclass case."""
    act_counts = get_act_counts(df)
    total_without_noops = 0
    for k, v in act_counts.items():
        if k != "NOOP":
            total_without_noops += v
    max_num_noops = int(total_without_noops * max_fraction)
    num_to_remove = act_counts["NOOP"] - max_num_noops
    if num_to_remove <= 0:
        return df

    all_noop_indices = df.index[df["NOOP"] == 1]
    noop_indices_to_remove = all_noop_indices[np.random.randint(len(all_noop_indices), size=num_to_remove)]

    df = df.drop(index=noop_indices_to_remove)
    return df


def balance_noops_multidiscrete(df, max_fraction):
    """Balances the dataset by action counts in the multidiscrete case."""
    all_noop_indices = []
    total_without_noops = 0
    for i, act_cols in df[["button", "y_axis", "x_axis"]].iterrows():
        if action_formatters.SSB64MultiDiscreteActionFormatter.is_noop(*tuple(act_cols)):
            all_noop_indices.append(i)
        else:
            total_without_noops += 1

    max_num_noops = int(total_without_noops * max_fraction)
    num_noops = len(all_noop_indices)
    num_to_remove = num_noops - max_num_noops
    if num_to_remove <= 0:
        return df

    all_noop_indices = np.array(all_noop_indices)
    absolute_indices_to_remove = np.random.randint(len(all_noop_indices), size=num_to_remove)
    noop_indices_to_remove = all_noop_indices[absolute_indices_to_remove]
    df = df.drop(index=noop_indices_to_remove)
    return df


def balance_noops(df, max_fraction, action_format):
    """Balance the fraction of noops in a n-frame dataset.

    Balancing noops probably yields worse performance than does simply weighting
    other actions in the loss (we do this during training as well), so long as
    it's used in conjunction with a per-batch balanced sampling. The problem
    though is that not rebalancing means you have to deal with potentially
    a lot more data, and when dealing with images that can slow things down.

    Args:
        df: The dataframe to balance, containing the actions.
        max_fraction: The max fraction of noops, beyond which samples are removed.

    Returns:
        The balanced dataframe.
    """
    assert max_fraction >= 0
    if action_format == "multiclass":
        return balance_noops_multiclass(df, max_fraction)
    elif action_format == "multidiscrete":
        return balance_noops_multidiscrete(df, max_fraction)
    else:
        raise ValueError("Invalid action format: {}".format(action_format))


def preprocess_image(input_filepath, output_filepath, transforms, encoding):
    """Preprocess an image with a provided set of transforms"""
    img = cv2.imread(input_filepath, encoding)
    img = transforms(img)
    torch.save(img, output_filepath)


def preprocess_dataset(dataset_filepath,
                       img_dir,
                       output_filepath,
                       preprocess_dir,
                       overwrite=False,
                       transforms=get_image_transforms(),
                       encoding=cv2.IMREAD_COLOR):
    """Preprocess a dataset.

    This function preprocesses a dataset by applying a set of transforms and
    saving the images in the dataset as transformed .pth files. It also generates
    a new dataframe that contains the relative paths of the images in the 
    preprocessed directory.
    """
    # Collect the list of filepaths to convert.
    df = pd.read_csv(dataset_filepath)
    img_cols = [c for c in df.columns if "frame" in c]
    relative_filenames = []
    for c in img_cols:
        relative_filenames.extend(list(df[c]))
    relative_filenames = list(set(relative_filenames))

    # Preprocess and save the images.
    n_files = len(relative_filenames)
    for i, relative_filename in enumerate(relative_filenames):

        sys.stdout.write("\r{} / {}".format(i + 1, n_files))
        image_filepath = os.path.join(img_dir, relative_filename)

        assert os.path.exists(image_filepath), image_filepath

        output_dir = os.path.join(preprocess_dir, os.path.dirname(relative_filename))
        if not os.path.exists(output_dir):
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_tensor_filepath = os.path.join(output_dir,
                                              os.path.basename(relative_filename)).replace(IMG_EXT, ".pth")

        if overwrite or not os.path.exists(output_tensor_filepath):
            preprocess_image(image_filepath, output_tensor_filepath, transforms, encoding)

    # Create the modified dataframe.
    tensor_df = df
    for c in img_cols:
        tensor_df[c] = [f.replace(IMG_EXT, ".pth") for f in tensor_df[c]]

    tensor_df.to_csv(output_filepath, index=False)


def get_action_formatter(action_format):
    if action_format == "multiclass":
        return action_formatters.SSB64MulticlassActionFormatter()
    elif action_format == "multidiscrete":
        return action_formatters.SSB64MultiDiscreteActionFormatter()
    else:
        raise ValueError("Invalid action format: {}".format(action_format))


def format_match(match_dir, args):
    """Format a match into a dataset.
    
    Args:
        match_dir: Directory containing the match data.
        args: The cmd line args describe in this file.
    """
    print("Match dir: {}".format(match_dir))
    assert os.path.exists(match_dir), "{}".format(match_dir)

    match_key = os.path.split(os.path.normpath(match_dir))[-1]
    match_data = MatchData(match_dir)

    os.makedirs(args.dataset_dir, exist_ok=True)
    output_filepath = os.path.join(args.dataset_dir, "{}.csv".format(match_key))

    if args.overwrite_formatting or not os.path.exists(output_filepath):
        print("Formatting...")
        action_formatter = get_action_formatter(args.action_format)
        formatter = MultiframeDatasetFormatter(match_data, action_formatter)
        df = formatter.to_df()
        df.to_csv(output_filepath, index=False)

    if args.noop_max_dataset_fraction is not None:
        print("Reducing NOOPs...")
        df = pd.read_csv(output_filepath)
        df = balance_noops(df, args.noop_max_dataset_fraction, args.action_format)
        df.to_csv(output_filepath, index=False)

    if args.preprocess_dataset:
        print("Preprocessing...")
        img_dir = os.path.dirname(os.path.normpath(match_dir))
        assert os.path.exists(img_dir)
        preprocess_dataset_filepath = os.path.join(args.dataset_dir, "preprocessed_{}.csv".format(match_key))
        preprocess_dataset(output_filepath, img_dir, preprocess_dataset_filepath, args.dataset_dir,
                           args.overwrite_preprocessing)


def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset and data loader args.
    parser.add_argument('--match_dirs',
                        nargs="+",
                        type=str,
                        help="Directories of the recorded data from the matchs to format.",
                        required=True)
    parser.add_argument('--dataset_dir',
                        type=str,
                        help=("Directory to store datasets in general."
                              " The actual directory name will be inferred from the match directory."),
                        required=True)
    parser.add_argument('--action_format',
                        type=str,
                        help=("The action format to use {multiclass, multidiscrete}."),
                        default="multiclass")
    parser.add_argument('--preprocess_dataset',
                        action="store_true",
                        help="Whether to preprocess the dataset.")
    parser.add_argument('--overwrite_formatting',
                        action="store_true",
                        help="Whether to redo formatting if it already exists.")
    parser.add_argument('--overwrite_preprocessing',
                        action="store_true",
                        help="Whether to redo preprocessing if it already exists.")
    parser.add_argument('--noop_max_dataset_fraction',
                        type=float,
                        help=("If provided, balances noops such that they"
                              "constitute at most this fraction of the dataset."
                              "For example, .2 would mean at most 20% of the dataset is noops."),
                        required=False,
                        default=0.1)
    return parser


def main():
    args = get_parser().parse_args()
    assert os.path.exists(args.dataset_dir)
    for match_dir in args.match_dirs:
        format_match(match_dir, args)


if __name__ == "__main__":
    main()
