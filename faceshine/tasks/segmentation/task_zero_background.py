#############################################################################
#
#   Source from:
#   https://github.com/leonelhs/face-makeup.PyTorch
#   Forked from:
#   https://github.com/zllrunning/face-makeup.PyTorch
#   Reimplemented by: Leonel Hernández
#
##############################################################################
import numpy as np
import torch

from faceshine import array2image, image_to_tensor
from faceshine.tasks import Task


class TaskZeroBackground(Task):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', weights='DEFAULT')
        self.model.eval()

    def executeTask(self, image):
        image = array2image(image)
        input_tensor = image_to_tensor(image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # create a binary (black and white) mask of the profile foreground
        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        return np.where(mask, 255, background).astype(np.uint8)
