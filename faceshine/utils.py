import json

import PIL
import cv2
import numpy as np
from PIL.Image import Image
import torchvision.transforms as transforms


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
