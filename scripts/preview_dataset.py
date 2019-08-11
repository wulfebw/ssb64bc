"""Visualizes an n-frame dataset.

Only the non-preprocessed case is implemented.
"""
import argparse
import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from ssb64bc.formatting.utils import get_act_cols, get_frame_cols


def display_row(row, frame_keys, action_keys, img_dir):
    frame_filenames = list(row[frame_keys])
    actions = list(row[action_keys])

    fig = plt.figure(figsize=(8, 8))

    action_text = "Nonzero actions:\n"
    for k, v in zip(action_keys, row[action_keys]):
        if v != 0:
            action_text += "{}: {} ".format(k, v)

    fig.suptitle(action_text, fontsize=14)

    for i, frame_filename in enumerate(frame_filenames):
        frame_filepath = os.path.join(img_dir, frame_filename)
        plt.subplot(2, 2, i + 1)
        plt.imshow(mpimg.imread(frame_filepath))

    plt.show()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_filepath', type=str, help="Dataset file to preview.", required=True)
    parser.add_argument('--img_dir',
                        type=str,
                        help="Where the images are stored (should contain the match dir).",
                        required=True)
    return parser


def main():
    args = get_parser().parse_args()
    df = pd.read_csv(args.dataset_filepath)
    action_keys = get_act_cols(df)
    frame_keys = get_frame_cols(df)
    for i, row in df.iterrows():
        display_row(row, frame_keys, action_keys, args.img_dir)


if __name__ == "__main__":
    main()
