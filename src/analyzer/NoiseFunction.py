import numpy as np
from abc import abstractmethod, ABC
from typing import Union, List

class NoiseFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _compute_noise_on_sample(self, x: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def __call__(self, x: Union[np.ndarray, List[np.ndarray]],
                       y: Union[np.ndarray, List[np.ndarray]]) -> float:
        if isinstance(x, np.ndarray):
            xlist = [x]
        else:
            xlist = x

        if isinstance(y, np.ndarray):
            ylist = [y]
        else:
            ylist = y

        if len(xlist) != len(ylist):
            raise RuntimeError("Unequal number of tensors to compare!")

        # Appiattisce tutti i tensori e concatena
        left = np.concatenate([t.flatten() for t in xlist])
        right = np.concatenate([t.flatten() for t in ylist])

        # Chiama il metodo astratto della sottoclasse
        return self._compute_noise_on_sample(left, right)
    

    
class L1Norm(NoiseFunction) :
    def _compute_noise_on_sample(self, x : np.ndarray, y : np.ndarray) :
        return np.linalg.norm(x - y, ord=1)

class L1NormAvg(NoiseFunction) :
    def _compute_noise_on_sample(self, x : np.ndarray, y : np.ndarray) :
        return np.linalg.norm(x - y, ord=1) / len(x)

class L2Norm(NoiseFunction) :
    def _compute_noise_on_sample(self, x : np.ndarray, y : np.ndarray) :
        return np.linalg.norm(x - y, ord=2)

class L2NormAvg(NoiseFunction) :
    def _compute_noise_on_sample(self, x : np.ndarray, y : np.ndarray) :
        return np.linalg.norm(x - y, ord=2) / len(x)
    
class InfNorm(NoiseFunction) :
    def _compute_noise_on_sample(self, x : np.ndarray, y : np.ndarray) :
        return np.linalg.norm(x - y, ord=np.inf)

