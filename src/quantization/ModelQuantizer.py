import copy
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import TensorsData, create_calibrator
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
from onnxruntime.quantization.quantize import QuantConfig

from quantization.DataReader import DataReader


class OnnxModelQuantizer:

    def __init__(
        self, model_path: str, calibration_data: np.ndarray, providers: list[str]
    ):
        self.tensors_range = self._build_tensors_range(
            model_path, calibration_data, providers
        )
        self.loaded_model = load_model_with_shape_infer(Path(model_path))

    def get_loaded_model(
        self,
    ):
        return self.loaded_model

    def _build_tensors_range(
        self, model_path: str, calibration_data: np.ndarray, providers: list[str]
    ) -> tuple[onnx.ModelProto, TensorsData]:
        tensors_range = None
        with tempfile.NamedTemporaryFile(suffix=".onnx") as augmented_model_file:
            # augmented_model_path = model_path.replace(".onnx", "_augmented.onnx")

            calibrator = create_calibrator(
                model_path,
                augmented_model_path=augmented_model_file.name,
                use_external_data_format=False,
                extra_options={},  ## TODO. In our case we are using no calibration option
                providers=providers,
            )
            data_reader = DataReader(model_path, calibration_data)
            calibrator.collect_data(data_reader=data_reader)
            tensors_range = calibrator.compute_data()

        if tensors_range is None:
            raise RuntimeError("Failed to compute tensors range")
        return tensors_range

    def quantize_model(
        self, quantization_config: QuantConfig, extra_options: dict[str]
    ) -> onnx.ModelProto:

        model_copy = copy.deepcopy(self.loaded_model)

        quantized_model = QDQQuantizer(
            model=model_copy,
            per_channel=quantization_config.per_channel,
            reduce_range=quantization_config.reduce_range,
            weight_qType=quantization_config.weight_type,
            activation_qType=quantization_config.activation_type,
            tensors_range=self.tensors_range,
            nodes_to_quantize=quantization_config.nodes_to_quantize,
            nodes_to_exclude=quantization_config.nodes_to_exclude,
            op_types_to_quantize=quantization_config.op_types_to_quantize,
            extra_options=extra_options,
        ).quantize_model()

        return quantized_model
