from abc import ABC, abstractmethod

import numpy as np


class NoiseFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _compute_noise_on_result(
        self, orig_resh_tensor: np.ndarray, quant_resh_tensor: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        orig_dict: dict[str, np.ndarray],
        quant_dict: dict[str, np.ndarray],
    ) -> dict[str, float]:

        for key in orig_dict.keys():
            if orig_dict[key].shape != quant_dict[key].shape:
                raise RuntimeError(f"Unequal shapes to compare for key {key}!")

        noise_dict: dict[str, np.ndarray] = {}
        for key in orig_dict.keys():
            orig_tensor = orig_dict[key]
            quant_tensor = quant_dict[key]
            orig_resh_tensor = orig_tensor.reshape((orig_tensor.shape[0], -1))
            quant_resh_tensor = quant_tensor.reshape((quant_tensor.shape[0], -1))

            result_noise_array = self._compute_noise_on_result(
                orig_resh_tensor, quant_resh_tensor
            )

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


class SignalNoiseRatio(NoiseFunction):
    def _compute_noise_on_result(
        self, orig_resh_tensor: np.ndarray, quant_resh_tensor: np.ndarray
    ):
        norm_ratio = np.linalg.norm(orig_resh_tensor, ord=1, axis=1) / np.linalg.norm(
            orig_resh_tensor - quant_resh_tensor + 1e-9, ord=1, axis=1
        )

        return 20 * np.log10(norm_ratio)
