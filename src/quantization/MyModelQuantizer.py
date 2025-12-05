from __future__ import annotations

import copy
import logging
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import (
    create_calibrator,
    quantize_static,
    write_calibration_table,
)
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    TensorsData,
)
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import (
    QuantFormat,
    QuantizationMode,
    QuantType,
    load_model_with_shape_infer,
    model_has_pre_process_metadata,
    save_and_reload_model_with_shape_infer,
    update_opset_version,
)
from onnxruntime.quantization.quantize import QuantConfig, check_static_quant_arguments
from onnxruntime.quantization.registry import (
    QDQRegistry,
    QLinearOpsRegistry,
)

from quantization.MyMinMaxCalibrator import MyMinMaxCalibrater


class MyModelQuantizer:

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        calibration_data_reader: CalibrationDataReader,
        providers: list[str],
        quant_config: QuantConfig,
        quant_format=QuantFormat.QDQ,
        extra_options=None,
    ):
        extra_options = extra_options or {}

        self.__init_quantizer(
            model_input,
            calibration_data_reader,
            quant_format,
            quant_config.op_types_to_quantize,
            quant_config.per_channel,
            quant_config.reduce_range,
            quant_config.activation_type,
            quant_config.weight_type,
            quant_config.nodes_to_quantize,
            quant_config.nodes_to_exclude,
            quant_config.use_external_data_format,
            CalibrationMethod.MinMax,
            providers,
            extra_options,
        )
        pass

    def quantize_nodes(self, nodes_to_quantize=None, nodes_to_exclude=None):
        model_copy = copy.deepcopy(self.model_to_quantize)
        quantizer = QDQQuantizer(
            model_copy,
            self.per_channel,
            self.reduce_range,
            self.weight_qType,
            self.activation_qType,
            self.tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            self.op_types_to_quantize,
            self.extra_options,
        )

        # quantizer = ONNXQuantizer(
        #     model_copy,
        #     self.per_channel,
        #     self.reduce_range,
        #     self.mode,
        #     True,
        #     self.weight_qType,
        #     self.activation_qType,
        #     self.tensors_range,
        #     nodes_to_quantize,
        #     nodes_to_exclude,
        #     self.op_types_to_quantize,
        #     self.extra_options,
        # )

        quantized_model = quantizer.quantize_model()
        onnx.checker.check_model(quantized_model)
        onnx.save_model(quantized_model, "quantized_model.onnx")

        raise Exception()

        del model_copy
        del quantizer

        return quantized_model

    def __init_quantizer(
        self,
        model_input: str | Path | onnx.ModelProto,
        calibration_data_reader: CalibrationDataReader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=None,
        per_channel=False,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        use_external_data_format=False,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=None,
        extra_options=None,
    ):

        if (
            activation_type == QuantType.QFLOAT8E4M3FN
            or weight_type == QuantType.QFLOAT8E4M3FN
        ):
            if calibrate_method != CalibrationMethod.Distribution:
                raise ValueError(
                    "Only Distribution calibration method is supported for float quantization."
                )

        extra_options = extra_options or {}
        nodes_to_exclude = nodes_to_exclude or []
        op_types_to_quantize = op_types_to_quantize or []
        mode = QuantizationMode.QLinearOps

        if not op_types_to_quantize or len(op_types_to_quantize) == 0:
            q_linear_ops = list(QLinearOpsRegistry.keys())
            qdq_ops = list(QDQRegistry.keys())
            op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

        model = (
            save_and_reload_model_with_shape_infer(model_input)
            if isinstance(model_input, onnx.ModelProto)
            else load_model_with_shape_infer(Path(model_input))
        )

        pre_processed: bool = model_has_pre_process_metadata(model)
        if not pre_processed:
            logging.warning(
                "Please consider to run pre-processing before quantization. Refer to example: "
                "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
                "/cpu/ReadMe.md "
            )

        calib_extra_options_keys = [
            ("CalibTensorRangeSymmetric", "symmetric"),
            ("CalibMovingAverage", "moving_average"),
            ("CalibMovingAverageConstant", "averaging_constant"),
            ("CalibMaxIntermediateOutputs", "max_intermediate_outputs"),
            ("CalibPercentile", "percentile"),
        ]
        calib_extra_options = {
            key: extra_options.get(name)
            for (name, key) in calib_extra_options_keys
            if name in extra_options
        }

        if extra_options.get("SmoothQuant", False):
            import importlib  # noqa: PLC0415

            try:
                importlib.import_module(
                    "neural_compressor.adaptor.ox_utils.smooth_quant"
                )
            except Exception as e:
                logging.error(f"{e}.")
                raise RuntimeError(
                    "neural-compressor is not correctly installed. Please check your environment."
                ) from e

            from neural_compressor.adaptor.ox_utils.smooth_quant import (  # noqa: PLC0415 # pyright: ignore[reportMissingImports]
                ORTSmoothQuant,
            )

            def inc_dataloader():
                data_reader = copy.deepcopy(calibration_data_reader)
                for data in data_reader:
                    yield data, None

            orig_nodes = [i.name for i in model.graph.node]
            dataloader = inc_dataloader()
            sq = ORTSmoothQuant(model_input, dataloader, reduce_range)
            del dataloader
            model = sq.transform(
                extra_options.get("SmoothQuantAlpha", 0.5),
                extra_options.get("SmoothQuantFolding", True),
            )
            sq_path = tempfile.TemporaryDirectory(prefix="ort.quant.")
            model_input = Path(sq_path.name).joinpath("sq_model.onnx").as_posix()
            model.save(model_input)
            nodes_to_exclude.extend(
                [i.name for i in model.model.graph.node if i.name not in orig_nodes]
            )
            model = load_model_with_shape_infer(
                Path(model_input)
            )  # use smooth quant model for calibration

        updated_model = update_opset_version(model, weight_type)
        is_model_updated = updated_model is not model
        if is_model_updated:
            model = updated_model

        with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
            if is_model_updated:
                # Update model_input and avoid to use the original one
                model_input = copy.deepcopy(model)

            if isinstance(model_input, onnx.ModelProto):
                output_path = (
                    Path(quant_tmp_dir).joinpath("model_input.onnx").as_posix()
                )
                onnx.save_model(
                    model_input,
                    output_path,
                    save_as_external_data=True,
                )
                model_input = output_path

            # calibrator = create_calibrator(
            #     model_input,
            #     op_types_to_quantize,
            #     augmented_model_path=Path(quant_tmp_dir)
            #     .joinpath("augmented_model.onnx")
            #     .as_posix(),
            #     calibrate_method=calibrate_method,
            #     use_external_data_format=use_external_data_format,
            #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            #     extra_options=calib_extra_options,
            # )

            calibrator = self.__create_calibrator(
                Path(model_input),
                op_types_to_quantize,
                nodes_to_quantize=nodes_to_quantize,
                augmented_model_path=Path(quant_tmp_dir)
                .joinpath("augmented_model.onnx")
                .as_posix(),
                calibrate_method=calibrate_method,
                use_external_data_format=use_external_data_format,
                providers=calibration_providers,
                extra_options=calib_extra_options,
            )

            stride = extra_options.get("CalibStridedMinMax", None)
            if stride:
                total_data_size = len(calibration_data_reader)
                if total_data_size % stride != 0:
                    raise ValueError(
                        f"Total data size ({total_data_size}) is not divisible by stride size ({stride})."
                    )

                for start in range(0, total_data_size, stride):
                    end_index = start + stride
                    calibration_data_reader.set_range(
                        start_index=start, end_index=end_index
                    )
                    calibrator.collect_data(calibration_data_reader)
            else:
                calibrator.collect_data(calibration_data_reader)
            tensors_range = calibrator.compute_data()
            # write_calibration_table(tensors_range)

            if not isinstance(tensors_range, TensorsData):
                raise TypeError(
                    f"Unexpected type {type(tensors_range)} for tensors_range and calibrator={type(calibrator)}."
                )
            del calibrator

        check_static_quant_arguments(quant_format, activation_type, weight_type)

        self.model_to_quantize = model
        self.per_channel = per_channel
        self.reduce_range = reduce_range
        self.mode = mode
        self.weight_qType = weight_type
        self.activation_qType = activation_type
        self.tensors_range = tensors_range
        self.nodes_to_exclude = nodes_to_exclude
        self.op_types_to_quantize = op_types_to_quantize
        self.extra_options = extra_options

    def __create_calibrator(
        self,
        model_path: Path,
        op_types_to_quantize: list[str],
        nodes_to_quantize: list[str],
        augmented_model_path: Path,
        calibrate_method: CalibrationMethod,
        use_external_data_format: bool,
        providers: list[str],
        extra_options: dict[str],
    ):
        calibrator = None
        if calibrate_method == CalibrationMethod.MinMax:
            # default settings for min-max algorithm
            symmetric = extra_options.get("symmetric", False)
            moving_average = extra_options.get("moving_average", False)
            averaging_constant = extra_options.get("averaging_constant", 0.01)
            max_intermediate_outputs = extra_options.get(
                "max_intermediate_outputs", None
            )
            per_channel = extra_options.get("per_channel", False)
            calibrator = MyMinMaxCalibrater(
                model_path,
                nodes_to_quantize,
                op_types_to_quantize,
                augmented_model_path,
                use_external_data_format=use_external_data_format,
                symmetric=symmetric,
                moving_average=moving_average,
                averaging_constant=averaging_constant,
                max_intermediate_outputs=max_intermediate_outputs,
                per_channel=per_channel,
            )
            if calibrator:
                calibrator.augment_graph()
                if providers:
                    calibrator.execution_providers = providers
                calibrator.create_inference_session()
                return calibrator

        raise ValueError(f"Unsupported calibration method {calibrate_method}")
