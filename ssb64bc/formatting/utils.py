import cv2
import pandas as pd

IMG_EXT = ".png"

def get_num_channels(image_type):
    if image_type == "color":
        return 3
    elif image_type == "grayscale":
        return 1
    else:
        raise ValueError("invalid image type {}".format(image_type))


def get_image_encoding(image_type):
    if image_type == "color":
        return cv2.IMREAD_COLOR
    elif image_type == "grayscale":
        return cv2.IMREAD_GRAYSCALE
    else:
        raise ValueError("invalid image type {}".format(args.image_type))

def get_frame_cols(df):
    """Returns the columns associated with frames."""
    return [c for c in df.columns if "frame" in c]

def get_act_cols(df):
    """Uses some simple (brittle) logic to figure out the action format,
    and returns the columns that correspond to actions.
    """
    if "button" in df.columns:
        # Multidiscrete case.
        return ["button", "y_axis", "x_axis"]
    elif "NOOP" in df.columns:
        # Assumes that only actions are all upper case.
        return [c for c in df.columns if c.upper() == c]
    else:
        raise ValueError("Action format not recognized; columns: {}".format(df.columns))


def get_act_counts(df):
    act_cols = get_act_cols(df)
    return dict(df[act_cols].sum(axis=0))


def get_num_actions(filepath):
    df = pd.read_csv(filepath)
    return len(get_act_cols(df))
