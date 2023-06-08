#############################################################################
#
#   Source from:
#   https://github.com/Li-Chongyi/Zero-DCE/
#   Forked from:
#   https://github.com/Li-Chongyi/Zero-DCE/
#   Reimplemented by: Leonel Hernández
#
##############################################################################
import logging
import os.path

import numpy as np
import torch
import torch.optim

from faceshine import array2image, tensor_to_ndarray
from faceshine.tasks import Task
from faceshine.tasks.lowlight.model import enhance_net_nopool
from huggingface_hub import snapshot_download

REPO_ID = "leonelhs/lowlight"
MODEL_NAME = "Epoch99.pth"


class TaskLowLight(Task):

    def __init__(self):
        super().__init__()
        self.model = enhance_net_nopool().cpu()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        snapshot_folder = snapshot_download(repo_id=REPO_ID)
        model_path = os.path.join(snapshot_folder, MODEL_NAME)
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)

    def executeTask(self, image):
        logging.info("Running low light enhancement")
        image = array2image(image)
        image = (np.asarray(image) / 255.0)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        image = image.cpu().unsqueeze(0)
        _, enhanced_image, _ = self.model(image)

        return tensor_to_ndarray(enhanced_image, nrow=8, padding=2, normalize=False)
