import cv2
import numpy as np
import torchvision

_COLOR_IMAGE_MEAN = np.array([90, 150, 120]) / 255.0
_GRAYSCALE_IMAGE_MEAN = np.array([120]) / 255.0


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


def get_image_transforms_for_encoding(encoding, resize=224, interpolation=0):
    image_type = "color" if encoding == cv2.IMREAD_COLOR else "grayscale"
    mean, std = get_image_mean_std(image_type)
    return get_image_transforms(mean, std, resize, interpolation)


def get_image_encoding(image_type):
    if image_type == "color":
        return cv2.IMREAD_COLOR
    elif image_type == "grayscale":
        return cv2.IMREAD_GRAYSCALE
    else:
        raise ValueError("invalid image type {}".format(args.image_type))
