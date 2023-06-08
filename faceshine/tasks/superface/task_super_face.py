import os

import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from huggingface_hub import snapshot_download
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from faceshine.tasks import Task

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

REALESRGAN_REPO_ID = 'leonelhs/realesrgan'
GFPGAN_REPO_ID = 'leonelhs/gfpgan'


def select_model(model_name):
    model = None
    netscale = 4
    dni_weight = None

    snapshot_folder = snapshot_download(repo_id=REALESRGAN_REPO_ID)
    model_path = os.path.join(snapshot_folder, model_name)

    if model_name == 'RealESRGAN_x4plus.pth':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    if model_name == 'RealESRNet_x4plus.pth':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    if model_name == 'RealESRGAN_x4plus_anime_6B.pth':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

    if model_name == 'RealESRGAN_x2plus.pth':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2  # This is

    if model_name == 'realesr-animevideov3.pth':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')

    if model_name == 'realesr-general-x4v3.pth':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model_path = [
            os.path.join(snapshot_folder, "realesr-general-wdn-x4v3.pth"),
            os.path.join(snapshot_folder, "realesr-general-x4v3.pth'")
        ]
        dni_weight = [0.2, 0.8]

    return model, netscale, model_path, dni_weight


class TaskSuperFace(Task):

    def __init__(self):
        super().__init__()

        scale = 2
        model_name = "RealESRGAN_x4plus.pth"
        model, netscale, model_path, dni_weight = select_model(model_name)
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=0)

        # output, _ = upsampler.enhance(img, outscale=outscale)

        snapshot_folder = snapshot_download(repo_id=GFPGAN_REPO_ID)
        model_path = os.path.join(snapshot_folder, "GFPGANv1.3.pth")
        self.face_enhancer = GFPGANer(
            model_path=model_path,
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    def executeTask(self, image):
        image = np.array(image)
        _, _, output = self.face_enhancer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
        return output
