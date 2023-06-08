#############################################################################
#
#   Source from:
#   https://github.com/leonelhs/face-makeup.PyTorch
#   Forked from:
#   https://github.com/zllrunning/face-makeup.PyTorch
#   Reimplemented by: Leonel Hernández
#
##############################################################################
import logging
import os.path

import torch
from PIL import Image

from faceshine.tasks import Task
from faceshine import image_to_tensor, array2image
from faceshine.tasks.faceparser import BiSeNet
from huggingface_hub import snapshot_download

REPO_ID = "leonelhs/faceparser"
MODEL_NAME = "79999_iter.pth"


class TaskFaceSegmentation(Task):
    def __init__(self):
        super().__init__()
        self.net = BiSeNet(n_classes=19)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        snapshot_folder = snapshot_download(repo_id=REPO_ID)
        model_path = os.path.join(snapshot_folder, MODEL_NAME)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.eval()

    """
        Predicts image segments needed to create an alpha blending mask. 
        Final mask will be created at client side using this process result
        :param image: Image file (jpg, png, ...)
        :returns: A numpy array.
        """

    def executeTask(self, image):
        logging.info("Running face segmentation.")
        with torch.no_grad():
            image = array2image(image)
            image = image.resize((512, 512), Image.BILINEAR)
            input_tensor = image_to_tensor(image)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            output = self.net(input_tensor)[0]
            return output.squeeze(0).cpu().numpy().argmax(0)
