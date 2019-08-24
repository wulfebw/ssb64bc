import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    # Dataset and data loader args.
    parser.add_argument('--filepaths', nargs="+", type=str, help="Dataset files to merge.", required=True)
    parser.add_argument('--output_filepath',
                        type=str,
                        help="Where to store the merged dataset file.",
                        required=True)
    return parser


def get_hdf5_info(filepaths):
    """Returns information about the hdf5 files provided."""
    assert len(filepaths) > 0
    num_samples = 0
    for filepath in filepaths:
        h5file = h5py.File(filepath, "r")
        num_samples += h5file["imgs"].shape[0]
        h5file.close()

    # Get the image and action dataset shapes.
    h5file = h5py.File(filepaths[0], "r")
    imgs_shape = list(h5file["imgs"].shape)
    imgs_shape[0] = num_samples
    imgs_shape = tuple(imgs_shape)

    actions_shape = list(h5file["actions"].shape)
    actions_shape[0] = num_samples
    actions_shape = tuple(actions_shape)

    attrs = dict(h5file.attrs)
    h5file.close()

    return dict(imgs_shape=imgs_shape, actions_shape=actions_shape, num_samples=num_samples, attrs=attrs)


def merge_hdf5_datasets(filepaths, output_filepath, chunk_size=100):
    info = get_hdf5_info(filepaths)
    h5file = h5py.File(output_filepath, "w")
    for k, v in info["attrs"].items():
        h5file.attrs[k] = v

    img_chunk_shape = list(info["imgs_shape"])
    img_chunk_shape[0] = chunk_size
    img_chunk_shape = tuple(img_chunk_shape)
    h5file.create_dataset("imgs", info["imgs_shape"], dtype=np.float32, chunks=img_chunk_shape)

    actions_dset = h5file.create_dataset("actions", info["actions_shape"], dtype=np.int)

    total_num_samples = info["num_samples"]
    sample_idx = 0
    for filepath in filepaths:
        cur_h5file = h5py.File(filepath, "r")
        num_samples = cur_h5file["imgs"].shape[0]
        for i in range(num_samples):
            sys.stdout.write("\r{} / {}".format(sample_idx + 1, total_num_samples))
            h5file["actions"][sample_idx] = cur_h5file["actions"][i]
            h5file["imgs"][sample_idx] = cur_h5file["imgs"][i]
            sample_idx += 1
        cur_h5file.close()
    h5file.close()


def merge_csv_datasets(filepaths):
    """Merge a list of csv datasets by simply cfoncatenating them."""
    df = pd.DataFrame()
    for filepath in filepaths:
        df = df.append(pd.read_csv(filepath))
    return df


def main():
    args = get_parser().parse_args()
    assert len(args.filepaths) > 1
    for filepath in args.filepaths:
        assert os.path.exists(filepath), "Filepath does not exist: {}".format(filepath)
    assert os.path.exists(os.path.dirname(args.output_filepath))

    ext = args.filepaths[0].split(".")[-1]
    if ext == "csv":
        df = merge_csv_datasets(args.filepaths)
        df.to_csv(args.output_filepath, index=False)
    elif ext == "hdf5":
        merge_hdf5_datasets(args.filepaths, args.output_filepath)
    else:
        raise ValueError("invalid file extension: {}".format(ext))


if __name__ == "__main__":
    main()
