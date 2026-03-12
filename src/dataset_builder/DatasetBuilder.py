import numpy as np
import pandas as pd
from tqdm import tqdm

from analyzer import AccuracyComputer, NoiseFunction
from analyzer.NoiseAnalyzer import NoiseAnalyzer


class DatasetBuilder:
    def __init__(self, noise_analyzer: NoiseAnalyzer):
        self.rng = np.random.default_rng(seed=0)  # generator con seed
        self.noise_analyzer: NoiseAnalyzer = noise_analyzer
        pass

    def build_dataset(
        self,
        blocks_num: int,
        dataset_size: int,
        noise_functions: (
            NoiseFunction.NoiseFunction | list[NoiseFunction.NoiseFunction]
        ) = None,
        accuracy_computer: AccuracyComputer.AccuracyFunction = None,
    ) -> dict:

        dataset_dict = {}

        if noise_functions is None:
            noise_functions = []
        elif isinstance(noise_functions, NoiseFunction.NoiseFunction):
            noise_functions = [noise_functions]
        else:
            noise_functions = noise_functions

        combinations_num = 2**blocks_num
        dataset_size = min(
            dataset_size - 2, combinations_num - 2
        )  ## Not considering the empty set and the full set

        extracted_nums = self.rng.choice(
            np.arange(1, combinations_num - 2), size=dataset_size, replace=False
        )
        extracted_nums = np.insert(extracted_nums, 0, 0)  ## Not quantized model
        extracted_nums = np.insert(
            extracted_nums, 1, combinations_num - 1
        )  ## Fully quantized model

        extracted_nums = np.sort(extracted_nums)

        n_bits = blocks_num

        for extracted_num in tqdm(extracted_nums):
            bit_array = ((extracted_num >> np.arange(n_bits - 1, -1, -1)) & 1).astype(
                np.uint8
            )
            curr_blocks_to_quantize = [i for i, bit in enumerate(bit_array) if bit == 1]

            curr_quant_results: dict[str, np.ndarray] = (
                self.noise_analyzer.compute_quantized_model_results(
                    curr_blocks_to_quantize
                )
            )
            accuracy_dict = self.noise_analyzer.compute_accuracy_on_results(
                curr_quant_results, accuracy_computer
            )

            accuracy_func_name = accuracy_computer.__class__.__name__
            if accuracy_func_name not in dataset_dict:
                dataset_dict[accuracy_func_name] = self.__init_dataset(
                    blocks_num, list(accuracy_dict.keys())
                )

            self.__insert_row(
                dataset_dict[accuracy_func_name],
                blocks_num,
                curr_blocks_to_quantize,
                accuracy_dict,
            )

            for noise_function in noise_functions:
                func_name = noise_function.__class__.__name__
                noise_dict = self.noise_analyzer.compute_noise_on_results(
                    curr_quant_results, noise_function
                )

                if func_name not in dataset_dict:
                    dataset_dict[func_name] = self.__init_dataset(
                        blocks_num, list(noise_dict.keys())
                    )

                self.__insert_row(
                    dataset_dict[func_name],
                    blocks_num,
                    curr_blocks_to_quantize,
                    noise_dict,
                )

        return dataset_dict

    def __init_dataset(self, blocks_num: int, target_names: list[str]):
        columns = [f"block_{i}" for i in range(blocks_num)] + target_names
        dataset = pd.DataFrame(columns=columns)
        return dataset

    def __insert_row(
        self,
        dataset: pd.DataFrame,
        blocks_num: int,
        curr_blocks_to_quantize: list[int],
        targets_dict: dict[str, float],
    ):
        new_row = {
            f"block_{block_idx}": (1 if block_idx in curr_blocks_to_quantize else 0)
            for block_idx in range(blocks_num)
        }
        new_row.update(targets_dict)
        dataset.loc[len(dataset)] = new_row
