import abc

import numpy as np


class Enhancer(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'enhance') and
                callable(subclass.enhance) or
                NotImplemented)

    @abc.abstractmethod
    def enhance(self, image=None, outscale=None) -> np.array:
        return NotImplementedError
