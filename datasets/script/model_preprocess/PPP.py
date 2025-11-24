from abc import ABC, abstractmethod
import numpy as np

class PPP(ABC):

    @abstractmethod
    def preprocess(self, original_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def postprocess(
        self,
        original_image: np.ndarray,
        predictions: np.ndarray,
        prototypes: np.ndarray,
        score_thr: float,
        iou_thr: float,
        num_classes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError