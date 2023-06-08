import PIL
import cv2
import json
import torch
import numpy as np
from PIL.Image import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid


def config():
    data = open('faceshine/conf.json')
    return json.load(data)


def array2image(ndarray):
    return PIL.Image.fromarray(np.uint8(ndarray)).convert('RGB')


def data2image(stream):
    np_img = np.fromstring(stream, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)


def image_to_tensor(image):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return to_tensor(image)


# Method are common to:
# TaskMaskScratches
# TaskEraseScratches
# TaskLowLight

def tensor_to_ndarray(tensor, nrow=1, padding=0, normalize=True):
    grid = make_grid(tensor, nrow, padding, normalize)
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


