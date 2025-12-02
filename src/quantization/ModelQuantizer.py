import copy
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import TensorsData, create_calibrator
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
from onnxruntime.quantization.quantize import QuantConfig
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

from quantization.DataReader import DataReader
from quantization.MyMinMaxCalibrator import MyMinMaxCalibrater


class OnnxModelQuantizer:

    def __init__(
        self,
        model_path: str,
        calib_dict: dict[str, list[np.ndarray]],
        providers: list[str],
        op_types_to_calibrate: list[str],
        nodes_to_calibrate: list[str],
        batch_size: int = 1,
    ):
        if not op_types_to_calibrate or len(op_types_to_calibrate) == 0:
            q_linear_ops = list(QLinearOpsRegistry.keys())
            qdq_ops = list(QDQRegistry.keys())
            self.op_types_to_calibrate = list(set(q_linear_ops + qdq_ops))
        else:
            self.op_types_to_calibrate = op_types_to_calibrate

        self.nodes_to_calibrate = nodes_to_calibrate
        self.model_path = model_path

        self.tensors_range = self._build_tensors_range(
            model_path,
            calib_dict,
            providers,
            nodes_to_calibrate,
            self.op_types_to_calibrate,
            batch_size,
        )
        self.loaded_model = load_model_with_shape_infer(Path(model_path))

    def get_loaded_model(
        self,
    ):
        return self.loaded_model

    def _build_tensors_range(
        self,
        model_path: str,
        calib_dict: dict[str, list[np.ndarray]],
        providers: list[str],
        nodes_to_calibrate: list[str],
        op_types_to_calibrate: list[str],
        batch_size: int,
    ) -> TensorsData:
        tensors_range = None
        with tempfile.NamedTemporaryFile(suffix=".onnx") as augmented_model_file:

            calibrator = self._create_calibrator(
                model_path,
                augmented_model_file.name,
                providers,
                nodes_to_calibrate,
                op_types_to_calibrate,
            )
            data_reader = DataReader(calib_dict, batch_size=batch_size)
            calibrator.collect_data(data_reader=data_reader)
            tensors_range = calibrator.compute_data()

        if tensors_range is None or not isinstance(tensors_range, TensorsData):
            raise RuntimeError("Failed to compute tensors range")
        return tensors_range

    def _create_calibrator(
        self,
        model_path,
        augmented_model_file_path,
        providers,
        nodes_to_calibrate,
        op_types_to_calibrate,
    ):
        calibrator = None
        calibrator = MyMinMaxCalibrater(
            model_path,
            nodes_to_calibrate,
            op_types_to_calibrate,
            augmented_model_path=augmented_model_file_path,
        )
        if calibrator:
            calibrator.augment_graph()
            if providers:
                calibrator.execution_providers = providers
            calibrator.create_inference_session()
        else:
            raise RuntimeError("Failed to create custom calibrator")

        # calibrator = create_calibrator(
        #     model_path,
        #     op_types_to_calibrate,
        #     augmented_model_file_path,
        #     providers=providers,
        # )

        return calibrator

    def quantize_model(
        self, quantization_config: QuantConfig, extra_options: dict[str]
    ) -> onnx.ModelProto:

        op_types_to_quantize = list(
            set(list(QLinearOpsRegistry.keys()) + list(QDQRegistry.keys()))
        )

        quantization_config.nodes_to_exclude = (
            []
            if quantization_config.nodes_to_exclude is None
            else quantization_config.nodes_to_exclude
        )
        quantization_config.nodes_to_quantize = (
            []
            if quantization_config.nodes_to_quantize is None
            else quantization_config.nodes_to_quantize
        )
        quantization_config.op_types_to_quantize = (
            op_types_to_quantize
            if quantization_config.op_types_to_quantize is None
            else quantization_config.op_types_to_quantize
        )

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

        del model_copy

        return quantized_model
