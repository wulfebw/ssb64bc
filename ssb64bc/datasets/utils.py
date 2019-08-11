import numpy as np
import torchvision

_COLOR_IMAGE_MEAN = np.array([90, 150, 120]) / 255.0
_GRAYSCALE_IMAGE_MEAN = np.array([120]) / 255.0


def get_hdf5_transfrom():
    return torchvision.transforms.ToTensor()


def get_image_mean_std(image_type):
    if image_type == "color":
        return _COLOR_IMAGE_MEAN, [1, 1, 1]
    elif image_type == "grayscale":
        return _GRAYSCALE_IMAGE_MEAN, [1]
    else:
        raise ValueError("invalid image type: {}".format(image_type))


def get_image_transforms(mean=_COLOR_IMAGE_MEAN, std=[1, 1, 1], resize=224, interpolation=0):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(resize, interpolation),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std, inplace=True)
    ])
