import abc


class Task(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'executeTask') and
                callable(subclass.executeTask) or
                NotImplemented)

    @abc.abstractmethod
    def executeTask(self, image):
        raise NotImplementedError
