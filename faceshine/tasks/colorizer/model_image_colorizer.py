import torch
from PIL import Image as PilImage

from deoldify.filters import IFilter, BaseFilter
from deoldify.visualize import ModelImageVisualizer
from fastai.basic_train import Learner
from fastai.vision import normalize_funcs

stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
results_dir = "./faceshine/tasks/colorizer/results"


class ImageFilter(BaseFilter):
    def __init__(self, learn: Learner):
        super().__init__(learn)
        self.render_base = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm, self.denorm = normalize_funcs(*stats)

    def filter(self, filtered_image: PilImage, render_factor=35) -> PilImage:
        orig_image = filtered_image.copy()
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)
        raw_color = self._unsquare(model_image, orig_image)
        return raw_color


class ModelImageColorizer(ModelImageVisualizer):
    def __init__(self, filter: IFilter):
        super().__init__(filter, results_dir=results_dir)

    def get_colored_image(self, image, render_factor: int = None) -> PilImage:
        self._clean_mem()
        return self.filter.filter(image, render_factor=render_factor)

