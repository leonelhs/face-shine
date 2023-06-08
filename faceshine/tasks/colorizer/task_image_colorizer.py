#############################################################################
#
#   Source from:
#   https://github.com/jantic/DeOldify
#   Forked from:
#   https://github.com/leonelhs/DeOldify
#   Reimplemented by: Leonel Hernández
#
##############################################################################
from pathlib import Path

import numpy as np
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.generators import gen_inference_deep
from huggingface_hub import snapshot_download

from faceshine import array2image
from faceshine.tasks import Task
from .model_image_colorizer import ImageFilter, ModelImageColorizer

device.set(device=DeviceId.CPU)

REPO_ID = "leonelhs/deoldify"
MODEL_NAME = "ColorizeArtistic_gen"


class TaskImageColorizer(Task):
    def __init__(self):
        super().__init__()
        snapshot_folder = snapshot_download(repo_id=REPO_ID)
        learn = gen_inference_deep(root_folder=Path(snapshot_folder), weights_name=MODEL_NAME)
        image_filter = ImageFilter(learn=learn)
        self.colorizer = ModelImageColorizer(image_filter)

    def executeTask(self, image):
        image = array2image(image)
        image = self.colorizer.get_colored_image(image, render_factor=35)
        return np.array(image)
