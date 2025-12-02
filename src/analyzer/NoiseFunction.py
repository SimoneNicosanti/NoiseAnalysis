from abc import ABC, abstractmethod

import numpy as np


class NoiseFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _compute_noise_on_result(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        x_dict: dict[str, np.ndarray],
        y_dict: dict[str, np.ndarray],
    ) -> dict[str, float]:

        for key in x_dict.keys():
            if x_dict[key].shape != y_dict[key].shape:
                raise RuntimeError(f"Unequal shapes to compare for key {key}!")

        noise_dict: dict[str, np.ndarray] = {}
        for key in x_dict.keys():
            x_tensor = x_dict[key]
            y_tensor = y_dict[key]
            left = x_tensor.reshape((x_tensor.shape[0], -1))
            right = y_tensor.reshape((y_tensor.shape[0], -1))

            result_noise_array = self._compute_noise_on_result(left, right)

            noise_dict[key] = float(np.mean(result_noise_array))

        return noise_dict


class L1Norm(NoiseFunction):
    def _compute_noise_on_result(self, x: np.ndarray, y: np.ndarray):
        return np.linalg.norm(x - y, ord=1, axis=1)


class L1NormAvg(NoiseFunction):
    def _compute_noise_on_result(self, x: np.ndarray, y: np.ndarray):
        return np.linalg.norm(x - y, ord=1, axis=1) / x.shape[1]


class L2Norm(NoiseFunction):
    def _compute_noise_on_result(self, x: np.ndarray, y: np.ndarray):
        return np.linalg.norm(x - y, ord=2, axis=1)


class L2NormAvg(NoiseFunction):
    def _compute_noise_on_result(self, x: np.ndarray, y: np.ndarray):
        return np.linalg.norm(x - y, ord=2, axis=1) / x.shape[1]


class InfNorm(NoiseFunction):
    def _compute_noise_on_result(self, x: np.ndarray, y: np.ndarray):
        return np.linalg.norm(x - y, ord=np.inf, axis=1)
