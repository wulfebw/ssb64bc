import argparse
import os

import cv2
import h5py
import imageio
import numpy as np


def decode_action(action_keys, action):
    if len(action_keys) == 3:
        # Multidiscrete case.
        action_text = ""
        for k, v in zip(action_keys, action):
            action_text += "{}: {} ".format(k.decode("utf-8"), v)
    else:
        # Multiclass case.
        action_text = action_keys[np.argmax(action)].decode("utf-8")

    return action_text


def generate_preview(dataset_filepath, output_dir):
    h5file = h5py.File(dataset_filepath, "r")
    images = h5file["imgs"]
    num_samples = len(images)

    idx = np.random.randint(num_samples)
    images = images[idx]
    actions = h5file["actions"][idx]

    text_x = 6
    text_y = 10
    text_images = []
    color = 0
    for action, image in zip(actions, images):
        if image.shape[-1] > 4:
            image = image.transpose(1, 2, 0)
        action_text = decode_action(h5file.attrs["action_keys"], action)
        text_images += [
            cv2.putText(img=image,
                        text=action_text,
                        org=(text_x, text_y),
                        fontFace=1,
                        fontScale=1,
                        color=color,
                        thickness=1).get()
        ]

    output_filepath = os.path.join(output_dir, "{}.gif".format(idx))
    imageio.mimsave(output_filepath, text_images)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_filepath', type=str, help="Dataset file to preview.", required=True)
    parser.add_argument('--preview_dir',
                        type=str,
                        help="Directory to output preview files to.",
                        required=True)
    parser.add_argument('--num_previews', type=int, default=5, help="Number of previews to generate.")
    return parser


def main():
    args = get_parser().parse_args()
    assert os.path.exists(args.dataset_filepath)
    assert os.path.exists(args.preview_dir)
    for _ in range(args.num_previews):
        generate_preview(args.dataset_filepath, args.preview_dir)


if __name__ == "__main__":
    main()
