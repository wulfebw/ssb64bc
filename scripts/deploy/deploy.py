#!/bin/python

import collections
import sys
import time

import gym
import gym_mupen64plus
import numpy as np
import torch

import deploy_utils as utils

sys.path.append("/Users/wulfebw/Programming/mupen64plus-mybuild/scripts")
import format_dataset
import datasets
import models


# mean = np.array([90, 150, 120]) / 255.0
mean = np.array([120]) / 255.0
std = [1]
transforms = datasets.get_image_transforms(mean, std)

# n_classes_per_action = (2, 2, 2, 2, 2, 2, 3, 3)
# model = models.MultiframeNonexclusiveActionModel(n_classes_per_action)
n_channels = 1
action_dim = 23
model = models.MultiframeMulticlassActionModel(output_dim=action_dim,
                                               n_channels=n_channels)
model = models.MultiframeMulticlassResnetActionModel(num_classes=action_dim)
# n_classes_per_action = format_dataset.SSB64MultiDiscreteActionFormatter.N_CLASSES
# model = models.MultiFrameMultiDiscreteActionModel(n_classes_per_action)
model.eval()

weights_filepath = "/Users/wulfebw/Programming/mupen64plus-mybuild/data/networks/regular_model_20.pth"
model.load_state_dict(torch.load(weights_filepath, map_location="cpu"))

# '''
# alright, this is getting ridiculous
# go to bed
# throw this away
# very far away in the morning

# turns out it worked after all so jokes on you
# '''
# whythefuckmap = {
#     0: 6 - 2,  # z
#     1: 3 - 2,  # b
#     2: 2 - 2,  # a
#     3: 4 - 2,  # rb
#     4: 5 - 2,  # lb
#     5: 7 - 2,  # c
# }

# def get_action_nonexclusive(x):
#     x = torch.cat(list(x), dim=0)
#     x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
#     probs = model.predict(x)

#     act = np.zeros(6, dtype=int)
#     sig_idxs = list(range(6))
#     for i, prob in enumerate(probs[:, sig_idxs].reshape(-1)):
#         print("prob {}: {}".format(i, prob))
#         if prob > np.random.rand():
#             act[whythefuckmap[i]] = 1
#         else:
#             act[whythefuckmap[i]] = 0

#     # append axis values to front
#     act = list(act)

#     # x axis
#     y_axis = np.argmax(probs[:, 9:])
#     print("y_probs: {}".format(probs[:, 9:]))
#     if y_axis == 0:
#         act = [0] + act
#     elif y_axis == 1:
#         act = [-120] + act
#     else:
#         act = [120] + act

#     # x axis
#     x_axis = np.argmax(probs[:, 6:9])
#     print("x_probs: {}".format(probs[:, 6:9]))
#     if x_axis == 0:
#         act = [0] + act
#     elif x_axis == 1:
#         act = [-120] + act
#     else:
#         act = [120] + act

#     return np.array(act).tolist()

def get_random_action():
    idx = np.random.randint(action_dim)
    return format_dataset.SSB64MulticlassActionFormatter.i2a(idx)

def get_action_exclusive(x, sample=True, sample_strat="multinomial"):
    x = torch.cat(list(x), dim=0)

    x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
    probs = np.array(model.predict(x)).reshape(-1)

    print(probs)

    if sample:
        if sample_strat == "multinomial":
            idx = utils.sample_probs(probs)
        elif sample_strat == "e-greedy":
            if np.random.rand() < 0.05:
                idx = np.random.randint(len(probs))
            idx = np.argmax(probs)
    else:
        idx = np.argmax(probs)

    # idx = np.argmax(np.array(probs).reshape(-1))

    print([round(p, 2) for p in np.array(probs).reshape(-1)])
    print(idx)

    return format_dataset.SSB64MulticlassActionFormatter.i2a(idx)


def get_action_multi_discrete(x):
    x = torch.cat(list(x), dim=0)
    x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
    probs = np.array(model.predict(x)).reshape(-1)

    edges = [0] + list(
        np.cumsum(format_dataset.SSB64MultiDiscreteActionFormatter.N_CLASSES).
        astype(int))
    idxs = []

    print()
    for start, end in zip(edges, edges[1:]):
        print(list(probs[start:end]))
        idx = utils.sample_probs(probs[start:end])
        idxs.append(idx)
    print(idxs)
    return format_dataset.SSB64MultiDiscreteActionFormatter.indices_to_action(
        idxs)


def main():
    env = gym.make('Smash-mario-v0')
    env.reset()
    
    utils.manual_navigate(env)
    # utils.navigate_to_practice(env)

    (obs, rew, end, info) = env.step([0, 0, 0, 0, 0, 0, 0, 0])

    d = collections.deque([], 4)
    for _ in range(4):
        d.append(transforms(obs))
    a = get_action_exclusive(d)
    # a = get_action_multi_discrete(d)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last_obs_timestep = time.time()
    avg_obs_timestep_diff = 0
    # now we fucking party
    for i in range(1000000):

        # a = get_action_nonexclusive(d)
        # a = get_action_exclusive(d)
        (obs, rew, end, info) = env.step(a)
        if i % 5 == 0:
            # x = torch.cat(list(d), dim=0)
            # x = x.unsqueeze(dim=1)
            # import torchvision
            # from PIL import ImageTk, Image

            # grid = torchvision.utils.make_grid(x, nrow=2, normalize=True)
            # grid = grid.numpy()
            # grid = grid.transpose(2,1,0)
            # plt.imshow(grid)
            # plt.savefig("/Users/wulfebw/Desktop/tmpimgs/{}.png".format(i))

            # plt.imshow(np.array(d[-1]).transpose(2,1,0)[:,:,0])
            # plt.savefig("/Users/wulfebw/Desktop/tmpimgs/{}.png".format(i))


        #    a = get_random_action()
            
            d.append(transforms(obs))
            a = get_action_exclusive(d,
                                     sample=False,
                                     sample_strat="multinomial")

            # a = get_action_multi_discrete(d)
            avg_obs_timestep_diff = avg_obs_timestep_diff * 0.95 + 0.05 * (
                time.time() - last_obs_timestep)
            last_obs_timestep = time.time()

        if i % 10 == 0:
            print("avg time between: ", avg_obs_timestep_diff)

    input("Press <enter> to exit... ")

    env.close()


if __name__ == "__main__":
    main()
