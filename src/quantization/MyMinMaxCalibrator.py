import itertools
from collections.abc import Sequence
from pathlib import Path

import onnx
from onnx import ModelProto, TensorProto
from onnxruntime.quantization.calibrate import MinMaxCalibrater


class MyMinMaxCalibrater(MinMaxCalibrater):
    def __init__(
        self,
        model_path: str | Path,
        nodes_to_calibrate: list[str],
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path="augmented_model.onnx",
        symmetric=False,
        use_external_data_format=False,
        moving_average=False,
        averaging_constant=0.01,
        max_intermediate_outputs=None,
        per_channel=False,
    ):
        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            symmetric=symmetric,
            use_external_data_format=use_external_data_format,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
            max_intermediate_outputs=max_intermediate_outputs,
            per_channel=per_channel,
        )
        self.model_path = model_path
        self.nodes_to_calibrate = nodes_to_calibrate
        self.op_types_to_calibrate = op_types_to_calibrate
        pass

    def select_tensors_to_calibrate(self, model: ModelProto):
        """
        select input/output tensors of candidate nodes to calibrate.
        returns:
            tensors (set): set of tensor name.
            value_infos (dict): tensor name to value info.
        """
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})
        initializer = {init.name for init in model.graph.initializer}

        tensors_to_calibrate = set()
        tensor_type_to_calibrate = {TensorProto.FLOAT, TensorProto.FLOAT16}

        for node in model.graph.node:
            if self._check_if_node_to_calibrate(node):
                for tensor_name in itertools.chain(node.input, node.output):
                    if tensor_name in value_infos:
                        vi = value_infos[tensor_name]
                        if (
                            vi.type.HasField("tensor_type")
                            and (
                                vi.type.tensor_type.elem_type
                                in tensor_type_to_calibrate
                            )
                            and (tensor_name not in initializer)
                        ):
                            tensors_to_calibrate.add(tensor_name)

        # tensors_to_calibrate.update([ot.name for ot in model.graph.output])
        # tensors_to_calibrate.update([it.name for it in model.graph.input])

        return tensors_to_calibrate, value_infos

    def _check_if_node_to_calibrate(self, node: onnx.NodeProto):
        return not self.nodes_to_calibrate or node.name in self.nodes_to_calibrate


def build_my_minmax_calibrator() -> MyMinMaxCalibrater:
    pass
