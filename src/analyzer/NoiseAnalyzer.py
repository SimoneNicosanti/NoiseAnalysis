import numpy as np
from quantization.ModelQuantizer import OnnxModelQuantizer
# from onnxruntime.quantization.qdq_loss_debug import compute_signal_to_quantization_noice_ratio
from onnxruntime.quantization.quantize import QuantConfig
import onnx

import onnxruntime as ort

from analyzer.NoiseFunction import NoiseFunction, L2Norm

class NoiseAnalyzer:
    def __init__(self, model_path: str, dataset_path: str, calib_size: int, eval_size: int, providers : list[str]):
        whole_dataset = np.load(dataset_path)
        dataset_list = [whole_dataset[key] for key in whole_dataset.files]

        self.calib_set = dataset_list[:calib_size]
        self.eval_set = dataset_list[calib_size:calib_size + eval_size]

        self.model_quantizer : OnnxModelQuantizer = OnnxModelQuantizer(model_path, self.calib_set, providers)
        self.original_results = self._compute_model_results(self.model_quantizer.get_loaded_model(), self.eval_set)
        pass

    def _compute_model_results(self, onnx_model : onnx.ModelProto, dataset: list[np.ndarray]) :
        model_results = []
        sess = ort.InferenceSession(onnx_model.SerializeToString())
        for input_elem in dataset :
            ## TODO We are assuming one input models
            result = sess.run(None, {sess.get_inputs()[0].name : input_elem})
            model_results.append(result)
            pass
        return model_results

    def _compute_noise_list(self, quantized_model_results : list, noise_function : NoiseFunction) :
        noises = []
        for idx, quantized_model_result in enumerate(quantized_model_results) :
            original_model_result = self.original_results[idx]
            sample_noise = noise_function(original_model_result, quantized_model_result)
            noises.append(sample_noise)
        
        return noises
    

    def compute_avg_noise(self, quantization_config : QuantConfig, extra_options : dict[str], noise_function : NoiseFunction = None) :
        if noise_function is None :
            noise_function = L2Norm()
        quantized_model : onnx.ModelProto = self.model_quantizer.quantize_model(quantization_config, extra_options)
        quantized_results = self._compute_model_results(quantized_model, self.eval_set)
        noises = self._compute_noise_list(quantized_results, noise_function)

        return float(np.mean(noises))

    