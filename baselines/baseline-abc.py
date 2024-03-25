from abc import ABC, abstractmethod

class Baseline(ABC):
    @abstractmethod
    def get_baseline(**kwargs):
        pass