from abc import ABC, abstractmethod


class ScoreLoader(ABC):
    @abstractmethod
    def read(self) -> None:
        pass

    @abstractmethod
    def downscale(self) -> None:
        pass
