"""Format a single, recurrent dataset.

The dataset takes the form of an h5py file containing sequences of
preprocessed images (as opposed to the n-frame case where the images
might not be preprocessed.)

We take the "stateless" recurrent training approach. See `preprocess_datasets`
for a description of this approach and its alternative.
"""
import os
import sys

import cv2
import h5py
import numpy as np
import pandas as pd

from ssb64bc.datasets.utils import get_image_transforms, get_image_mean_std
import ssb64bc.formatting.action_formatters as action_formatters
from ssb64bc.formatting.match_data import MatchData
from ssb64bc.formatting.multi_frame_dataset_formatter import MultiframeDatasetFormatter
from ssb64bc.formatting.utils import IMG_EXT, get_act_cols

import format_dataset
import utils


def get_image_shape():
    """Get the shape of a preprocessed image."""
    # TODO: Get the shape by applying the transform to the first image.
    return (1, 224, 298)


def get_num_samples(filepaths, max_seq_len):
    """Get the total number of samples that will result from converting the filepaths
    into recurrent samples of provided max_seq_len.
    """
    total_samples = 0
    for filepath in filepaths:
        cur_samples = len(pd.read_csv(filepath)) // max_seq_len
        total_samples += cur_samples
    return total_samples


def preprocess_datasets(filepaths, img_dir, args):
    """Preprocess the datasets into a single recurrent dataset.

    At this point, each dataframe contains a single match formatted as (frame, action) pairs.

    Now we need to convert this to a format that can be combined with sequences from other matches.
    There are two options:
    1. treat this entire match as a single sequence, and during training maintain the hidden state
    across BPTT iterations.
    2. break this match up into fixed length sequences and treat them independently, ignoring the
    empty hidden state at the beginning of the sequence.
    I think option (1) might achieve better performance in theory. In practice, option (2) is
    a fair amount easier to implement, so that's what I'm going with.

    To simplify the recurrent logic throughout, we format the dataset such that all sequences
    are the same length (args.max_seq_len) (throwing out timesteps on the end that don't make
    a full sample). In the case of full match sequences, this isn't a big loss.
    """
    print("Preprocessing...")
    assert len(filepaths) > 0
    assert args.overwrite_preprocessing or not os.path.exists(args.output_filepath)

    h5file = h5py.File(args.output_filepath, "w", libver='latest')

    num_samples = get_num_samples(filepaths, args.max_seq_len)

    transforms = get_image_transforms(*get_image_mean_std("grayscale"))
    images_shape = (num_samples, ) + (args.max_seq_len, ) + get_image_shape()
    print("Shape of images: {}".format(images_shape))
    chunk_shape = (min(100, num_samples), ) + (args.max_seq_len, ) + get_image_shape()
    images_dset = h5file.create_dataset("imgs", images_shape, dtype=np.float32, chunks=chunk_shape)

    action_keys = get_act_cols(pd.read_csv(filepaths[0]))
    h5file.attrs["action_keys"] = [s.encode("ascii", "ignore") for s in action_keys]
    actions_shape = (num_samples, ) + (args.max_seq_len, ) + (len(action_keys), )
    actions_dset = h5file.create_dataset("actions", actions_shape, dtype=np.int)

    n_filepaths = len(filepaths)
    sample_idx = 0
    for i, filepath in enumerate(filepaths):
        df = pd.read_csv(filepath)
        n_timesteps = len(df)
        for timestep, (index, row) in enumerate(df.iterrows()):
            sys.stdout.write("\rtimestep: {} / {} match: {} / {}".format(timestep + 1, n_timesteps, i + 1,
                                                                         n_filepaths))
            timestep_idx = timestep % args.max_seq_len
            if timestep_idx == 0 and timestep != 0:
                sample_idx += 1
                # Ignore partial samples.
                if sample_idx >= num_samples:
                    break

            image_filepath = os.path.join(img_dir, row["frame_0"])
            images_dset[sample_idx, timestep_idx, ...] = utils.preprocess_image(image_filepath,
                                                                                transforms).numpy()
            actions_dset[sample_idx, timestep_idx, ...] = list(row[action_keys])

    h5file.close()


def add_recurrent_arguments(parser):
    parser.add_argument('--output_filepath',
                        type=str,
                        help=("Filepath to output preprocessed dataset file."
                              " This is required because in the recurrent case matches are combined."),
                        required=True)
    parser.add_argument('--max_seq_len',
                        type=int,
                        help=("The maximum length of a sample in the dataset."
                              "The match will be broken up into samples of this length."),
                        required=False,
                        default=50)
    return parser


def main():
    parser = format_dataset.get_parser()
    parser = add_recurrent_arguments(parser)
    args = parser.parse_args()
    assert os.path.exists(args.dataset_dir)
    # Set the number of frames per action to 1 since we operate over full sequences.
    args.n_frames = 1

    formatted_dataset_filepaths = []
    img_dir = None
    for match_dir in args.match_dirs:
        cur_img_dir = os.path.dirname(os.path.normpath(match_dir))
        # Make sure the image directory is the same for each match.
        assert img_dir is None or cur_img_dir == img_dir
        img_dir = cur_img_dir
        formatted_dataset_filepaths += [format_dataset.format_match(match_dir, args)]
    assert img_dir is not None

    # Always preprocess in the recurrent case.
    preprocess_datasets(formatted_dataset_filepaths, img_dir, args)


if __name__ == "__main__":
    main()
