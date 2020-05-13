import os

import cv2
import torch


def match_key_from_match_dir(match_dir):
    """Converts the match directory into a key for the match."""
    return os.path.split(os.path.normpath(match_dir))[-1]


def preprocess_image(input_filepath, transforms=None, encoding=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(input_filepath, encoding)
    img = transforms(img)
    # Intentionally do not transform to (H, W, C)
    return img


def preprocess_and_save_image(input_filepath,
                              output_filepath,
                              transforms=None,
                              encoding=cv2.IMREAD_GRAYSCALE):
    """Preprocess an image with a provided set of transforms"""
    img = preprocess_image(input_filepath, transforms, encoding)
    torch.save(img, output_filepath)


def get_cv2_encoding_from_string(string):
    if string == "color":
        return cv2.IMREAD_COLOR
    elif string == "grayscale":
        return cv2.IMREAD_GRAYSCALE
    else:
        raise ValueError(f"Invalid encoding string: {string}")
