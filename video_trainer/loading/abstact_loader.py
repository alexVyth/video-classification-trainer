from abc import ABC
from abc import abstractmethod


class ScoreLoader(ABC):

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def downscale(self):
        pass
