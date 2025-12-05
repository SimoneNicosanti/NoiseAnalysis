from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto
from onnxruntime.quantization import StaticQuantConfig, quantize


class ConditionalModelBuilder:

    def __init__(
        self,
    ):

        pass

    def build_conditional_model(
        self,
        model_input: str | Path | onnx.ModelProto,
        static_quant_config: StaticQuantConfig,
    ):

        with tempfile.NamedTemporaryFile() as temp_file:
            quantize(model_input, temp_file.name, static_quant_config)
            quantized_model = onnx.load(temp_file.name)

        original_model = onnx.load(model_input)

        mixed_model = self.__build_mixed_model(
            original_model, quantized_model, static_quant_config.nodes_to_quantize
        )

        del original_model

        return mixed_model

    def __build_mixed_model(
        self,
        original_model: onnx.ModelProto,
        quantized_model: onnx.ModelProto,
        nodes_to_quantize: list[str],
    ) -> onnx.ModelProto:

        mixed_model = copy.deepcopy(original_model)

        original_model = onnx.shape_inference.infer_shapes(original_model)
        quantized_model = onnx.shape_inference.infer_shapes(quantized_model)

        not_quant_extractor = onnx.utils.Extractor(original_model)
        quant_extractor = onnx.utils.Extractor(quantized_model)

        not_quant_init_names = [
            tensor.name for tensor in original_model.graph.initializer
        ]

        for node_name in nodes_to_quantize:

            original_node, node_idx = self.__find_original_node(
                original_model, node_name
            )
            extract_input = [
                tensor_name
                for tensor_name in original_node.input
                if tensor_name not in not_quant_init_names
            ]
            extract_output = original_node.output

            not_quant_block = not_quant_extractor.extract_model(
                extract_input, extract_output
            )

            extract_output = [
                tensor_name + "_DequantizeLinear_Output"
                for tensor_name in extract_output
            ]
            quant_block = quant_extractor.extract_model(extract_input, extract_output)

            if_node, cond_tensor = self.__build_if_node(
                not_quant_block, quant_block, original_node
            )

            mixed_model = self.__integrate_if_node(
                mixed_model, if_node, cond_tensor, node_idx
            )

        del original_model
        del quantized_model

        return mixed_model

        pass

    def __integrate_if_node(
        self,
        mixed_model: onnx.ModelProto,
        if_node: onnx.NodeProto,
        cond_tensor: onnx.TensorProto,
        node_idx: int,
    ) -> onnx.ModelProto:

        mixed_model.graph.node.pop(node_idx)
        mixed_model.graph.node.insert(node_idx, if_node)
        mixed_model.graph.input.append(cond_tensor)

        onnx.checker.check_model(mixed_model)

        return mixed_model

        pass

    def __find_original_node(
        self, model: onnx.ModelProto, node_name: str
    ) -> tuple[onnx.NodeProto, int]:

        for idx, node in enumerate(model.graph.node):
            if node.name == node_name:
                return node, idx
        return None, None

    def __build_if_node(
        self,
        not_quant_block: onnx.ModelProto,
        quant_block: onnx.ModelProto,
        node: onnx.NodeProto,
    ):

        node_name = node.name

        not_quant_block = self.__fix_block_tensors_names(
            not_quant_block, ".else_branch"
        )
        quant_block = self.__fix_block_tensors_names(quant_block, ".then_branch")

        if_graph = onnx.helper.make_graph(
            nodes=quant_block.graph.node,
            name=f"Branch_{node_name}_is_quant",
            inputs=[],  # <-- tensore dichiarato come input
            outputs=quant_block.graph.output,
            initializer=quant_block.graph.initializer,
            value_info=quant_block.graph.value_info,
        )

        else_graph = onnx.helper.make_graph(
            nodes=not_quant_block.graph.node,
            name=f"Branch_{node_name}_is_not_quant",
            inputs=[],  # <-- tensore dichiarato come input
            outputs=not_quant_block.graph.output,
            initializer=not_quant_block.graph.initializer,
            value_info=not_quant_block.graph.value_info,
        )

        cond_name = f"cond_{node_name}_is_quant"

        cond_input = onnx.helper.make_tensor_value_info(cond_name, TensorProto.BOOL, [])
        if_node = onnx.helper.make_node(
            "If",
            inputs=[cond_name],
            outputs=node.output,
            name=f"If_{node_name}_is_quant",
            then_branch=if_graph,
            else_branch=else_graph,
        )

        return if_node, cond_input

    def __fix_block_tensors_names(self, block: onnx.ModelProto, suffix: str):
        block_input_names = [tens.name for tens in block.graph.input]
        block_output_names = [tens.name for tens in block.graph.output]

        for tens in block.graph.initializer:
            if (
                tens.name not in block_input_names
                and tens.name not in block_output_names
            ):
                tens.name = tens.name + suffix

        for tens in block.graph.value_info:
            if (
                tens.name not in block_input_names
                and tens.name not in block_output_names
            ):
                tens.name = tens.name + suffix

        for node in block.graph.node:
            for i, out_name in enumerate(node.output):
                if (
                    out_name not in block_input_names
                    and out_name not in block_output_names
                ):
                    node.output[i] = out_name + suffix
            for i, in_name in enumerate(node.input):
                if (
                    in_name not in block_input_names
                    and in_name not in block_output_names
                ):
                    node.input[i] = in_name + suffix
            pass

        onnx.checker.check_model(block)

        return block
