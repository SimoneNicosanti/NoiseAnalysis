from abc import ABC, abstractmethod

import numpy as np


class PPP(ABC):

    @abstractmethod
    def preprocess(self, original: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        original_input: dict[str, np.ndarray],
        output: dict[str, np.ndarray],
        **kwargs,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError
