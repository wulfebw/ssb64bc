import functools
import glob
import os
import pathlib
import sys

import pandas as pd

from ssb64bc.formatting.utils import IMG_EXT


@functools.total_ordering
class ImgFile:
    """A single image file and associated timestamp."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.timestamp = ImgFile._parse_timestamp(filepath)

    @property
    def key_filepath(self):
        return os.path.join(*pathlib.Path(self.filepath).parts[-3:])

    @staticmethod
    def _parse_timestamp(filepath):
        basename = os.path.basename(filepath)
        return int(basename.replace(IMG_EXT, ""))

    def __eq__(self, other):
        return self.timestamp == other.timestamp and self.filepath == other.filepath

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __hash__(self):
        return hash((self.timestamp, self.filepath))

    def __repr__(self):
        return "(filepath: {} timestamp: {})".format(self.filepath, self.timestamp)


class MatchData:
    """This class represents the raw data from a match.

    This class loads in the image filepaths and actions, and provides methods for accessing them.
    """
    # When the data is recorded, the images are stored in this directory,
    # and actions in a file with this name
    IMGS_DIR_NAME = "imgs"
    ACTIONS_FILENAME = "output.csv"

    def __init__(self, match_directory):
        imgs_directory = os.path.join(match_directory, self.IMGS_DIR_NAME)
        assert os.path.exists(imgs_directory), "Images directory does not exist: {}".format(imgs_directory)
        self.img_files = MatchData._load_img_filepaths(imgs_directory, IMG_EXT)

        actions_filepath = os.path.join(match_directory, self.ACTIONS_FILENAME)
        assert os.path.exists(actions_filepath), "Actions filepath does not exist: {}".format(
            actions_filepath)
        self.actions = MatchData._load_actions(actions_filepath)

    @staticmethod
    def _load_img_filepaths(directory, ext):
        """Load images from the provided directory into `ImgFile` objects."""
        pattern = os.path.join(directory, "*{}".format(ext))
        filepaths = glob.glob(pattern)
        return sorted([ImgFile(f) for f in filepaths])

    @staticmethod
    def _load_actions(filepath):
        """Load the actions as a DataFrame."""
        return pd.read_csv(filepath)

    def _action_is_null(self, action):
        """Return True if the provided action is null.

        An action is null if all entries in it are zeros.
        """
        return not any([a != 0 for a in action[self.actions.columns != 'timestamp']])

    def imgs_and_actions(self, max_time_ahead=0.05, n_frames=1):
        """Yields (img, action) pairs, grouping them based on timestamp.

        Each image is associated with an action by selecting the first 
        non-null action that occurs after the image within some timeframe (max_time_ahead),
        subject to the constraint that both the image and action are unique.
        - If only an empty / noop action is found within the timeframe, then it is returned.

        Args:
            max_time_ahead: Seconds ahead to search for the relevant action.
            n_frames: The number of sequential frames to return.
        """
        assert n_frames >= 1
        max_time_ahead_microseconds = max_time_ahead * 1e6
        n_imgs = len(self.img_files)
        n_actions = len(self.actions)
        img_idx = n_frames - 1
        action_idx = 0
        while img_idx < n_imgs:
            img = self.img_files[img_idx]

            action = None
            while action_idx < n_actions:
                next_action = self.actions.iloc[action_idx]

                if next_action.timestamp < img.timestamp:
                    # If the action occurs before the image, skip this action.
                    action_idx += 1
                    continue

                if next_action.timestamp - img.timestamp > max_time_ahead_microseconds:
                    # If the action occurs after the max time ahead, skip the image.
                    break

                if not self._action_is_null(next_action):
                    # If we find an action that's non-null, break and use it.
                    action = next_action
                    break
                elif action is None:
                    # We found a valid, but null action. Store it, but keep looking.
                    action = next_action

                # Move to the next action.
                action_idx += 1

            if action is not None:
                # Yield the previous n_frames-1 frames as well.
                yield (self.img_files[img_idx - n_frames + 1:img_idx + 1], action)
                # Move forward the number of stacked frames.
                img_idx += n_frames
            else:
                print("No action found for image index {}".format(img_idx))
                # Move to only the next image in this case.
                img_idx += 1
