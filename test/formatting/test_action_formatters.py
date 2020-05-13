import tempfile
import unittest

import numpy as np
import pandas as pd

import ssb64bc.formatting.action_formatters as action_formatters


class MockAction:

    COLUMNS = [
        "R_DPAD", "L_DPAD", "D_DPAD", "U_DPAD", "START_BUTTON", "Z_TRIG", "B_BUTTON", "A_BUTTON",
        "R_CBUTTON", "L_CBUTTON", "D_CBUTTON", "U_CBUTTON", "R_TRIG", "L_TRIG", "X_AXIS", "Y_AXIS",
        "timestamp"
    ]

    def __init__(self, high_cols, axes={}, timestamp=0):
        action = [0] * len(self.COLUMNS) + [timestamp]
        for i, col in enumerate(self.COLUMNS):
            action[i] = 1 if col in high_cols else 0
        for axis, val in axes.items():
            index = self.COLUMNS.index(axis)
            action[index] = val
        self.row = pd.DataFrame(data=[action], columns=self.COLUMNS + ["timestamp"]).iloc[0]


class TestSSB64MulticlassActionFormatter(unittest.TestCase):
    def test_action_to_index(self):
        formatter = action_formatters.SSB64MulticlassActionFormatter()
        input_index_pairs = [
            (["A_BUTTON"], 3),
            (["Z_TRIG"], 1),
            ((["A_BUTTON"], {
                "Y_AXIS": 120
            }), 15),
            ((["A_BUTTON"], {
                "X_AXIS": 120
            }), 16),
            ((["A_BUTTON"], {
                "Y_AXIS": -120
            }), 17),
            ((["A_BUTTON"], {
                "X_AXIS": -120
            }), 18),
        ]
        for inputs, expected_index in input_index_pairs:
            actual_index = np.argmax(formatter(MockAction(*inputs).row))
            self.assertEqual(actual_index, expected_index)


if __name__ == '__main__':
    unittest.main()
