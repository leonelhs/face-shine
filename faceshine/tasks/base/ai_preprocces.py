import abc

import numpy as np


class Preprocess(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'enhance') and
                callable(subclass.enhance) or
                NotImplemented)

    @abc.abstractmethod
    def process(self, image=None) -> np.array:
        return NotImplementedError
