import numpy as np


def sample_probs(probs):
    probs = np.reshape(probs, -1)
    if np.sum(probs.astype(np.float64)) > 1.0:
        return np.random.randint(len(probs))
    return np.argmax(np.random.multinomial(1, probs, 1))


def wait(env, timesteps=50):
    for _ in range(timesteps):
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0, 0, 0, 0])


def navigate_to_practice(env):
    wait(env)

    # press some a's
    for _ in range(1):
        (obs, rew, end, info) = env.step([0, 0, 1, 0, 0, 0, 0, 0])
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0, 0, 0, 0])

    wait(env)
    # press some a's
    for _ in range(1):
        (obs, rew, end, info) = env.step([0, 0, 1, 0, 0, 0, 0, 0])
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0, 0, 0, 0])

    wait(env)

    # press some a's
    for _ in range(10):
        (obs, rew, end, info) = env.step([0, 0, 1, 0, 0, 0, 0, 0])
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0, 0, 0, 0])

    wait(env)
    # go down the menu
    for _ in range(1):
        (obs, rew, end, info) = env.step([0, -80, 0, 0, 0, 0, 0, 0])

    # wait(env)
    # press some a's
    for _ in range(1):
        (obs, rew, end, info) = env.step([0, 0, 1, 0, 0, 0, 0, 0])
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0, 0, 0, 0])

    wait(env)
    wait(env)
    # go up
    for _ in range(20):
        (obs, rew, end, info) = env.step([0, 80, 0, 0, 0, 0, 0, 0])

    # wait(env)
    # go right
    for _ in range(10):
        (obs, rew, end, info) = env.step([80, 0, 0, 0, 0, 0, 0, 0])

    # wait(env)
    # press some a's
    for _ in range(1):
        (obs, rew, end, info) = env.step([0, 0, 1, 0, 0, 0, 0, 0])
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0, 0, 0, 0])

    # wait(env)

    press_start_env = env
    while not hasattr(press_start_env, "press_start"):
        assert hasattr(press_start_env, "env")
        press_start_env = press_start_env.env

    for _ in range(5):
        press_start_env.press_blue_dk()
    for _ in range(5):
        press_start_env.press_start()

    wait(env)

    # go down the menu
    for _ in range(5):
        (obs, rew, end, info) = env.step([0, -80, 0, 0, 0, 0, 0, 0])

    wait(env)

    # go right
    for _ in range(5):
        (obs, rew, end, info) = env.step([80, 0, 0, 0, 0, 0, 0, 0])

    # wait(env)
    # press some a's
    for _ in range(10):
        (obs, rew, end, info) = env.step([0, 0, 1, 0, 0, 0, 0, 0])

    wait(env)


def manual_navigate(env):

    press_start_env = env
    while not hasattr(press_start_env, "press_start"):
        assert hasattr(press_start_env, "env")
        press_start_env = press_start_env.env

    def decode_char(ch):
        if ch == "w":  # up
            return [0, 80, 0, 0, 0, 0, 0, 0]
        if ch == "d":  # right
            return [80, 0, 0, 0, 0, 0, 0, 0]
        if ch == "s":  # down
            return [0, -80, 0, 0, 0, 0, 0, 0]
        if ch == "a":  # left
            return [-80, 0, 0, 0, 0, 0, 0, 0]
        if ch == "g":
            return [0, 0, 1, 0, 0, 0, 0, 0]
        if ch == "b":
            press_start_env.press_blue_dk()
            return [0, 0, 0, 0, 0, 0, 0, 0]
        if ch == "v":
            press_start_env.press_start()
            return [0, 0, 0, 0, 0, 0, 0, 0]
        if ch == "q":
            return None
        return [0, 0, 0, 0, 0, 0, 0, 0]

    while True:
        ch = input()
        cmd = decode_char(ch)
        if cmd is None:
            return
        for _ in range(3):
            env.step(cmd)
