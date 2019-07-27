import collections
import sys

import pandas as pd


class MultiframeDatasetFormatter:
    """A multi-frame dataset formatter.

    Each sample consists of:
    1. input: N img filepaths to stack and use as input. 
        Note that these are stored in the dataset as the actual filepath.
    2. target: a single action associated with the stack of imgs.
        The format of the action depends on the action-formatter used.
    """

    def __init__(self, match_data, action_formatter, n_frames=4):
        """
        Args:
            match_data: A MatchData object.
            action_formatter: An object that formats actions.
            n_frames: The number of historical frames to include in each sample.
                For recurrent datasets, this should be a single frame.
        """
        self.action_formatter = action_formatter
        self.n_frames = n_frames
        self.dataset = self._format_dataset(match_data)

    def _format_dataset(self, match_data):
        dataset = collections.OrderedDict()
        sample_img_files = collections.deque([], self.n_frames)
        for i, (img_files, action) in enumerate(match_data.imgs_and_actions(n_frames=1)):
            sys.stdout.write("\r{}".format(i + 1))
            sample_img_files.append(img_files[0])
            if len(sample_img_files) < self.n_frames:
                continue
            key = tuple(img_file.key_filepath for img_file in sample_img_files)
            dataset[key] = self.action_formatter(action)
        return dataset

    def to_df(self):
        """Converts the internal `MatchData` to a pandas DataFrame."""
        # Get the columns of the dataframe.
        file_columns = ["frame_{}".format(i) for i in range(self.n_frames)]
        action_columns = self.action_formatter.labels
        columns = file_columns + action_columns

        # Get the rows.
        data = []
        for (k, v) in self.dataset.items():
            row = k + v
            data.append(row)

        data = sorted(data)
        df = pd.DataFrame(data=data, columns=columns)
        df.index.name = "sample_id"
        return df
