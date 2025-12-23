import numpy as np
import pandas as pd
from onnxruntime.quantization.quantize import QuantConfig
from tqdm import tqdm

from analyzer import NoiseFunction
from analyzer.NoiseAnalyzer import NoiseAnalyzer


class DatasetBuilder:
    def __init__(self, noise_analyzer: NoiseAnalyzer):
        self.rng = np.random.default_rng(seed=0)  # generator con seed
        self.noise_analyzer: NoiseAnalyzer = noise_analyzer
        pass

    def build_dataset(
        self,
        nodes_names: list[str],
        dataset_size: int,
        noise_functions=None,
    ):

        dataset_dict = {}

        if noise_functions is None:
            noise_functions = [NoiseFunction.L2Norm()]

        if isinstance(noise_functions, NoiseFunction.NoiseFunction):
            noise_functions = [noise_functions]

        combinations_num = 2 ** len(nodes_names)
        dataset_size = min(
            dataset_size - 1, combinations_num - 2
        )  ## Not considering the empty set and the full set

        extracted_nums = self.rng.choice(
            np.arange(1, combinations_num - 2), size=dataset_size, replace=False
        )
        extracted_nums = np.insert(extracted_nums, 0, combinations_num - 1)

        n_bits = len(nodes_names)

        for extracted_num in tqdm(extracted_nums):
            bit_array = ((extracted_num >> np.arange(n_bits - 1, -1, -1)) & 1).astype(
                np.uint8
            )
            curr_nodes_names = [
                nodes_names[i] for i, bit in enumerate(bit_array) if bit == 1
            ]

            curr_quant_results: dict[str, np.ndarray] = (
                self.noise_analyzer.compute_quantized_model_results(curr_nodes_names)
            )

            for noise_function in noise_functions:
                func_name = noise_function.__class__.__name__
                noise_dict = self.noise_analyzer.compute_noise_on_results(
                    curr_quant_results, noise_function
                )

                if func_name not in dataset_dict:
                    columns = nodes_names + list(noise_dict.keys())
                    dataset = pd.DataFrame(columns=columns)
                    dataset_dict[func_name] = dataset

                new_row = {
                    node_name: 1 if node_name in curr_nodes_names else 0
                    for node_name in nodes_names
                }
                new_row.update(noise_dict)
                dataset_dict[func_name].loc[len(dataset_dict[func_name])] = new_row

        return dataset_dict
