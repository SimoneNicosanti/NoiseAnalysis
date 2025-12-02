import numpy as np
import onnx
import onnxruntime as ort

# from onnxruntime.quantization.qdq_loss_debug import compute_signal_to_quantization_noice_ratio
from onnxruntime.quantization.quantize import QuantConfig

from analyzer.NoiseFunction import L2Norm, NoiseFunction
from quantization.ModelQuantizer import OnnxModelQuantizer


class NoiseAnalyzer:
    def __init__(
        self,
        model_path: str,
        dataset_dict: dict[str, np.ndarray],
        calib_size: int,
        eval_size: int,
        providers: list[str],
        op_types_to_calibrate: list[str],
        nodes_to_calibrate: list[str],
        batch_size: int,
    ):
        input_names = [input_name for input_name in dataset_dict.keys()]

        self.calib_dict = {
            input_name: dataset_dict[input_name][:calib_size]
            for input_name in input_names
        }
        self.eval_dict = {
            input_name: dataset_dict[input_name][calib_size : calib_size + eval_size]
            for input_name in input_names
        }
        self.providers = providers
        self.batch_size = batch_size

        self.model_quantizer: OnnxModelQuantizer = OnnxModelQuantizer(
            model_path,
            self.calib_dict,
            self.providers,
            op_types_to_calibrate,
            nodes_to_calibrate,
            self.batch_size,
        )

        self.original_model = self.model_quantizer.get_loaded_model()

        self.original_results = self._compute_model_results(
            self.original_model, self.eval_dict, self.batch_size
        )

        pass

    def _compute_model_results(
        self,
        onnx_model: onnx.ModelProto,
        dataset: dict[str, np.ndarray],
        batch_size: int,
    ) -> dict[str, np.ndarray]:

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=self.providers
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

    def compute_avg_noise(
        self,
        quantization_config: QuantConfig,
        extra_options: dict[str],
        noise_functions: list[NoiseFunction] | NoiseFunction = None,
    ):
        if noise_functions is None:
            noise_functions = [L2Norm()]

        if isinstance(noise_functions, NoiseFunction):
            noise_functions = [noise_functions]

        quantized_model: onnx.ModelProto = self.model_quantizer.quantize_model(
            quantization_config, extra_options
        )
        quantized_results = self._compute_model_results(
            quantized_model, self.eval_dict, self.batch_size
        )

        function_dict = {}
        for noise_function in noise_functions:
            function_dict[noise_function.__class__.__name__] = noise_function.__call__(
                self.original_results, quantized_results
            )

        return function_dict
