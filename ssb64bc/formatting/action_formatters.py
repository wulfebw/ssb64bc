class SSB64MulticlassActionFormatter:
    """A multi-class action formatter for SSB64.

    This means the actions are formulated as a one-hot vector.
    And different in-game actions (e.g., A + left) are formatted as a single action.
    """

    KEEP_KEYS = [
        "Z_TRIG", "B_BUTTON", "A_BUTTON", "R_CBUTTON", "L_CBUTTON", "D_CBUTTON", "U_CBUTTON", "R_TRIG",
        "L_TRIG", "X_AXIS", "Y_AXIS"
    ]
    NON_C_NON_AXIS_KEYS = ["Z_TRIG", "B_BUTTON", "A_BUTTON", "R_TRIG", "L_TRIG"]
    AXIS_THRESHOLD = 25

    def __init__(self):
        self.labels = []

        # Do nothing.
        self.labels += ["NOOP"]

        # Individual buttons.
        self.all_buttons = ["Z_TRIG", "B_BUTTON", "A_BUTTON", "R_TRIG", "L_TRIG", "C_BUTTON"]
        self.labels += self.all_buttons

        # Individual directions.
        self.directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        self.labels += self.directions

        # Certain buttons + directionals.
        self.directional_buttons = ["Z_TRIG", "A_BUTTON", "B_BUTTON"]
        for button in self.directional_buttons:
            for direction in self.directions:
                self.labels += ["{}+{}".format(button, direction)]

        self.n_classes = len(self.labels)

    def _get_direction(self, action):
        """Returns the directional of the action or None if no direction is selected.

        If multiple directions are active (though the axes), the larger one is returned.
        If a c button is pressed, then up is always returned.
        """
        c = any(action[["R_CBUTTON", "L_CBUTTON", "D_CBUTTON", "U_CBUTTON"]] != 0)
        if c:
            return "UP"

        x = action["X_AXIS"]
        y = action["Y_AXIS"]

        if abs(x) < self.AXIS_THRESHOLD and abs(y) < self.AXIS_THRESHOLD:
            return None
        elif abs(x) > abs(y):
            return "LEFT" if x < 0 else "RIGHT"
        else:
            return "UP" if y > 0 else "DOWN"

    def _get_index(self, action):
        # Only keep the set of keys that are relevant for ssb64.
        action = action[self.KEEP_KEYS]

        # Get direction of the action, if any.
        direction = self._get_direction(action)

        # Remove c buttons and axes at this point.
        action = action[self.NON_C_NON_AXIS_KEYS]

        # Determine how many buttons are nonzero.
        n_buttons_pressed = sum(action != 0)

        if n_buttons_pressed == 0 and direction is None:
            # No op.
            return 0
        elif n_buttons_pressed == 0 and direction is not None:
            return self.labels.index(direction)
        elif n_buttons_pressed == 1 and direction is None:
            # Must be a single button.
            for button in self.all_buttons:
                if action[button] != 0:
                    return self.labels.index(button)
        elif n_buttons_pressed == 1 and direction is not None:
            # A pair consisting of a single button and direction.
            for button in self.directional_buttons:
                if action[button] != 0:
                    return self.labels.index("{}+{}".format(button, direction))
            # Reaching this point means it was a non-directional button + a direction.
            # Just return the direction in this case.
            return self.labels.index(direction)
        else:
            # Ignore everything else for now.
            return 0

    def __call__(self, action):
        # Format the action as a one-hot vector to conform to the format expected by the dataset formatters.
        formatted = [0] * self.n_classes
        idx = self._get_index(action)
        formatted[idx] = 1
        return tuple(formatted)

    @staticmethod
    def i2a(idx):
        i2a_map = {
            0: [0, 0, 0, 0, 0, 0, 0, 0],  # noop
            1: [0, 0, 0, 0, 0, 0, 1, 0],  # z
            2: [0, 0, 0, 1, 0, 0, 0, 0],  # b
            3: [0, 0, 1, 0, 0, 0, 0, 0],  # a
            4: [0, 0, 0, 0, 1, 0, 0, 0],  # rtrig
            5: [0, 0, 0, 0, 0, 1, 0, 0],  # ltrig
            6: [0, 0, 0, 0, 0, 0, 0, 1],  # c
            7: [0, 120, 0, 0, 0, 0, 0, 0],  # up
            8: [120, 0, 0, 0, 0, 0, 0, 0],  # right
            9: [0, -120, 0, 0, 0, 0, 0, 0],  # down
            10: [-120, 0, 0, 0, 0, 0, 0, 0],  # left
            11: [0, 120, 0, 0, 0, 0, 1, 0],  # ztrig + up
            12: [120, 0, 0, 0, 0, 0, 1, 0],  # ztrig + right
            13: [0, -120, 0, 0, 0, 0, 1, 0],  # ztrig + down
            14: [-120, 0, 0, 0, 0, 0, 1, 0],  # ztrig + left
            15: [0, 120, 1, 0, 0, 0, 0, 0],  # a + up
            16: [120, 0, 1, 0, 0, 0, 0, 0],  # a + right
            17: [0, -120, 1, 0, 0, 0, 0, 0],  # a + down
            18: [-120, 0, 1, 0, 0, 0, 0, 0],  # a + left
            19: [0, 120, 0, 1, 0, 0, 0, 0],  # b + up 
            20: [120, 0, 0, 1, 0, 0, 0, 0],  # b + right
            21: [0, -120, 0, 1, 0, 0, 0, 0],  # b + down
            22: [-120, 0, 0, 1, 0, 0, 0, 0],  # b + left
        }
        return i2a_map[idx]


class SSB64MultiDiscreteActionFormatter:
    """Formats actions as three discrete classes.

    1. The button pressed
        - Options: z, a, b, r_trig, nothing (noop)
        - 5 classes
    2. The y-axis
        - This is derived from the joystick, but the c-buttons are also converted to this
        - All the c buttons map to up
        - 3 classes (down, nothing, up)
    3. The x-axis
       - This is only the joystick x axis
       - 3 classes again (left, nothing, right)
    """
    # These are the keys that can be converted to the button entry.
    # This ordering defines their precedence if multiple are pressed at once.
    BUTTON_KEYS = ["A_BUTTON", "B_BUTTON", "R_TRIG", "Z_TRIG"]

    # The number of buttons expected by the gym environment.
    N_GYM_BUTTONS = 6
    # Maps the key to its index in the list of buttons in the action used in the gym environment.
    BUTTON2INDEX = {
        "A_BUTTON": 0,
        "B_BUTTON": 1,
        "R_TRIG": 2,
        "Z_TRIG": 4,
    }

    # Axis constants.
    AXIS_THRESHOLD = 40
    AXIS_ABS_MAX = 125
    C_KEYS = ["R_CBUTTON", "L_CBUTTON", "D_CBUTTON", "U_CBUTTON"]
    Y_AXIS_KEY = "Y_AXIS"
    X_AXIS_KEY = "X_AXIS"

    # Reporting constants.
    LABELS = ["button", "x_axis", "y_axis"]
    N_CLASSES = [5, 3, 3]

    def __init__(self):
        self.labels = SSB64MultiDiscreteActionFormatter.LABELS

    @staticmethod
    def _button_action_to_index(action):
        """"Returns the integer index of the class of button pressed."""
        for i, button_key in enumerate(SSB64MultiDiscreteActionFormatter.BUTTON_KEYS):
            if int(action[button_key]) == 1:
                return i
        # Reaching this point indicates a noop.
        return len(SSB64MultiDiscreteActionFormatter.BUTTON_KEYS)

    @staticmethod
    def _button_index_to_action(idx):
        """Converts the index to the one-hot vector button action."""
        action = [0] * SSB64MultiDiscreteActionFormatter.N_GYM_BUTTONS

        # If idx points to a key we consider, then set it.
        # If this condition isn't true, it's a noop.
        if idx < len(SSB64MultiDiscreteActionFormatter.BUTTON_KEYS):
            key = SSB64MultiDiscreteActionFormatter.BUTTON_KEYS[idx]
            action_idx = SSB64MultiDiscreteActionFormatter.BUTTON2INDEX[key]
            action[action_idx] = 1

        return action

    @staticmethod
    def _y_axis_index_to_action(idx):
        if idx == 0:
            return 0
        elif idx == 1:
            return SSB64MultiDiscreteActionFormatter.AXIS_ABS_MAX
        else:
            return -SSB64MultiDiscreteActionFormatter.AXIS_ABS_MAX

    @staticmethod
    def _y_axis_action_to_index(action):
        """Returns the y axis class.
        0 = nothing
        1 = up
        2 = down
        """
        # All c keys indicate up.
        for c_key in SSB64MultiDiscreteActionFormatter.C_KEYS:
            if int(action[c_key]) == 1:
                return 1

        y_val = action[SSB64MultiDiscreteActionFormatter.Y_AXIS_KEY]
        if y_val < -SSB64MultiDiscreteActionFormatter.AXIS_THRESHOLD:
            return 2
        elif y_val > SSB64MultiDiscreteActionFormatter.AXIS_THRESHOLD:
            return 1
        return 0

    @staticmethod
    def _x_axis_index_to_action(idx):
        if idx == 0:
            return 0
        elif idx == 1:
            return SSB64MultiDiscreteActionFormatter.AXIS_ABS_MAX
        else:
            return -SSB64MultiDiscreteActionFormatter.AXIS_ABS_MAX

    @staticmethod
    def _x_axis_action_to_index(action):
        """Returns the x axis class.
        0 = nothing
        1 = right
        2 = left
        """
        x_val = action[SSB64MultiDiscreteActionFormatter.X_AXIS_KEY]
        if x_val < -SSB64MultiDiscreteActionFormatter.AXIS_THRESHOLD:
            return 2
        elif x_val > SSB64MultiDiscreteActionFormatter.AXIS_THRESHOLD:
            return 1
        return 0

    @staticmethod
    def action_to_indices(action):
        """Converts the action to a set of three discrete actions."""
        formatted = []
        formatted += [SSB64MultiDiscreteActionFormatter._button_action_to_index(action)]
        formatted += [SSB64MultiDiscreteActionFormatter._y_axis_action_to_index(action)]
        formatted += [SSB64MultiDiscreteActionFormatter._x_axis_action_to_index(action)]
        return tuple(formatted)

    def __call__(self, action):
        return SSB64MultiDiscreteActionFormatter.action_to_indices(action)

    @staticmethod
    def indices_to_action(indices):
        """Converts class indices to the action format."""
        action = []
        action += [SSB64MultiDiscreteActionFormatter._x_axis_index_to_action(indices[2])]
        action += [SSB64MultiDiscreteActionFormatter._y_axis_index_to_action(indices[1])]
        action += SSB64MultiDiscreteActionFormatter._button_index_to_action(indices[0])
        return action
