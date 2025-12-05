import numpy as np
import onnx
import onnxruntime as ort

# from onnxruntime.quantization.qdq_loss_debug import compute_signal_to_quantization_noice_ratio
from onnxruntime.quantization.quantize import QuantConfig, QuantFormat

from analyzer.NoiseFunction import NoiseFunction
from quantization.DataReader import DataReader
from quantization.MyModelQuantizer import MyModelQuantizer


class NoiseAnalyzer:
    def __init__(
        self,
        model_path: str,
        dataset_dict: dict[str, np.ndarray],
        calib_size: int,
        eval_size: int,
        providers: list[str],
        batch_size: int,
        quant_config: QuantConfig,
        quant_format=QuantFormat.QDQ,
        extra_options=None,
    ):
        input_names = [input_name for input_name in dataset_dict.keys()]

        self.calib_dict = {
            input_name: dataset_dict[input_name][:calib_size]
            for input_name in input_names
        }
        self.providers = providers
        self.batch_size = batch_size

        data_reader = DataReader(self.calib_dict, batch_size=batch_size)

        self.model_quantizer: MyModelQuantizer = MyModelQuantizer(
            model_path,
            data_reader,
            self.providers,
            quant_config,
            quant_format,
            extra_options,
        )

        self.eval_dict = {
            input_name: dataset_dict[input_name][calib_size : calib_size + eval_size]
            for input_name in input_names
        }

        original_model = onnx.load_model(model_path)
        self.original_results = self._compute_model_results(
            original_model, self.eval_dict, self.batch_size
        )
        del original_model

        pass

    def _compute_model_results(
        self,
        onnx_model: onnx.ModelProto,
        dataset: dict[str, np.ndarray],
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = (
            3  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
        )
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=self.providers,
            sess_options=sess_options,
        )

        output_names = [out.name for out in sess.get_outputs()]
        model_results = {out_name: [] for out_name in output_names}

        data_size = list(dataset.values())[0].shape[0]

        for elem_idx in range(0, data_size, batch_size):
            input_dict = {}
            for input_name in dataset.keys():
                input_numpy_array = dataset[input_name][
                    elem_idx : elem_idx + batch_size
                ]
                input_dict[input_name] = input_numpy_array

            result_list = sess.run(output_names, input_dict)
            for out_idx, output_name in enumerate(model_results.keys()):
                model_results[output_name].append(result_list[out_idx])

        for out_name in model_results.keys():
            model_results[out_name] = np.concatenate(model_results[out_name], axis=0)

        del sess

        return model_results

    def compute_quantized_model_results(
        self,
        nodes_to_quantize: tuple[str],
    ) -> dict[str, np.ndarray]:

        quantized_model: onnx.ModelProto = self.model_quantizer.quantize_nodes(
            nodes_to_quantize
        )
        quantized_results = self._compute_model_results(
            quantized_model, self.eval_dict, self.batch_size
        )

        return quantized_results

    def compute_noise_on_results(
        self, quantized_results: dict[str, np.ndarray], noise_function: NoiseFunction
    ):

        return noise_function.__call__(self.original_results, quantized_results)
