import numpy as np
import onnx
import onnxruntime as ort

# from onnxruntime.quantization.qdq_loss_debug import compute_signal_to_quantization_noice_ratio
from onnxruntime.quantization.quantize import StaticQuantConfig

from analyzer.NoiseFunction import NoiseFunction
from quantization.ConditionalModelBuilder import ConditionalModelBuilder
from quantization.DataReader import DataReader


class NoiseAnalyzer:
    def __init__(
        self,
        model_path: str,
        dataset_dict: dict[str, np.ndarray],
        calib_size: int,
        eval_size: int,
        providers: list[str],
        batch_size: int,
        static_quant_config: StaticQuantConfig,
    ):
        input_names = [input_name for input_name in dataset_dict.keys()]

        self.calib_dict = {
            input_name: dataset_dict[input_name][:calib_size]
            for input_name in input_names
        }
        self.providers = providers
        self.batch_size = batch_size

        static_quant_config.calibration_data_reader = DataReader(
            self.calib_dict, batch_size=batch_size
        )

        self.model_quantizer: ConditionalModelBuilder = ConditionalModelBuilder()
        conditional_model = self.model_quantizer.build_conditional_model(
            model_path, static_quant_config
        )

        self.sess = self.__build_inference_session(conditional_model)

        self.eval_dict = {
            input_name: dataset_dict[input_name][calib_size : calib_size + eval_size]
            for input_name in input_names
        }

        self.all_quantizable_nodes = static_quant_config.nodes_to_quantize
        curr_quant_dict = {node_name: False for node_name in self.all_quantizable_nodes}
        self.original_results = self._compute_model_results(
            self.eval_dict, self.batch_size, curr_quant_dict
        )

        pass

    def __build_inference_session(self, model: onnx.ModelProto):
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = (
            3  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
        )
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=self.providers,
            sess_options=sess_options,
        )

        return sess

    def _compute_model_results(
        self,
        dataset: dict[str, np.ndarray],
        batch_size: int,
        quant_dict: dict[str, bool],
    ) -> dict[str, np.ndarray]:

        output_names = [out.name for out in self.sess.get_outputs()]
        model_results = {out_name: [] for out_name in output_names}

        data_size = list(dataset.values())[0].shape[0]

        for elem_idx in range(0, data_size, batch_size):
            input_dict = {}
            for input_name in dataset.keys():
                input_numpy_array = dataset[input_name][
                    elem_idx : elem_idx + batch_size
                ]
                input_dict[input_name] = input_numpy_array

            for quant_layer_key, quant_layer_value in quant_dict.items():
                input_name = f"cond_{quant_layer_key}_is_quant"
                input_dict[input_name] = np.array(quant_layer_value, dtype=np.bool_)

            result_list = self.sess.run(output_names, input_dict)
            for out_idx, output_name in enumerate(model_results.keys()):
                model_results[output_name].append(result_list[out_idx])

        for out_name in model_results.keys():
            model_results[out_name] = np.concatenate(model_results[out_name], axis=0)

        return model_results

    def compute_quantized_model_results(
        self,
        curr_nodes_to_quantize: tuple[str],
    ) -> dict[str, np.ndarray]:

        quant_dict = {
            node_name: True if node_name in curr_nodes_to_quantize else False
            for node_name in self.all_quantizable_nodes
        }

        quantized_results = self._compute_model_results(
            self.eval_dict, self.batch_size, quant_dict
        )

        return quantized_results

    def compute_noise_on_results(
        self, quantized_results: dict[str, np.ndarray], noise_function: NoiseFunction
    ):
        return noise_function.__call__(self.original_results, quantized_results)
