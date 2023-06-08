#############################################################################
#
#   Source from:
#   https://github.com/leonelhs/zeroscratches
#   Forked from:
#   https://github.com/leonelhs/zeroscratches
#   Reimplemented by: Leonel Hernández
#
##############################################################################

from zeroscratches import EraseScratches

from faceshine import array2image
from faceshine.tasks import Task


class TaskEraseScratches(Task):

    def __init__(self):
        super().__init__()
        self.scratchEraser = EraseScratches()

    def executeTask(self, image):
        image = array2image(image)
        return self.scratchEraser.erase(image)

