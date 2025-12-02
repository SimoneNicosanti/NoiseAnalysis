import numpy as np
from onnxruntime.quantization.quantize import QuantConfig

from analyzer.NoiseAnalyzer import NoiseAnalyzer
from analyzer.NoiseFunction import NoiseFunction


class DatasetBuilder:
    def __init__(self):
        self.list_rng = np.random.default_rng(seed=0)  # generator con seed
        self.len_rng = np.random.default_rng(seed=1)
        pass

    def build_dataset(
        self,
        noise_analyzer: NoiseAnalyzer,
        nodes_names: list[str],
        nodes_types: list[str],
        test_set_size: int,
        train_set_size: int,
        noise_functions: list[NoiseFunction] | NoiseFunction = None,
    ):

        tot_set_size = test_set_size + train_set_size
        used_configs = set()

        while len(used_configs) < tot_set_size:
            curr_list_len = self.len_rng.integers(1, len(nodes_names) + 1)
            curr_nodes_names = tuple(
                sorted(
                    self.list_rng.choice(nodes_names, size=curr_list_len, replace=False)
                )
            )
            if curr_nodes_names in used_configs:
                continue

            quant_config = QuantConfig(
                nodes_to_quantize=curr_nodes_names,
                op_types_to_quantize=nodes_types,
            )

            avg_noises = noise_analyzer.compute_avg_noise(
                quantization_config=quant_config,
                extra_options={},
                noise_functions=noise_functions,
            )
            print(curr_nodes_names)
            print(avg_noises)

            used_configs.add(curr_nodes_names)

        pass
