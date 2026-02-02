from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxslim
from onnx import TensorProto


class ConditionalModelBuilder:

    block_format_string = "block_{block_idx}_is_quant"

    def __init__(
        self,
    ):

        pass

    def build_conditional_model(
        self,
        orig_model_input: str | Path | onnx.ModelProto,
        quant_model_input: str | Path | onnx.ModelProto,
        blocks_num: int,
    ):

        original_model = onnx.load_model(orig_model_input)
        quantized_model = onnx.load_model(quant_model_input)

        mixed_model = self.__build_mixed_model(
            original_model, quantized_model, blocks_num
        )

        # execution_wrapper = ExecutionWrapper(
        #     nq_extracted_models=nq_extracted_models,
        #     q_extracted_models=q_extracted_models,
        # )

        del original_model

        return mixed_model

    def __find_valid_quantized_blocks(
        self, original_graph: nx.DiGraph, original_topo_sort: list, parts
    ):
        quantized_blocks = []

        blocks_extremities = []
        for part in parts:
            first_node_idx = part[0]
            last_node_idx = part[-1]

            blocks_extremities.append((first_node_idx, last_node_idx))

        for couple_idx, couple in enumerate(blocks_extremities):
            block_nodes = original_topo_sort[couple[0] : couple[1] + 1]

            for block_node in block_nodes:
                ## TODO Make this dynamic !!!
                curr_node = original_graph.nodes[block_node]["node"]
                if curr_node is not None and curr_node.op_type in [
                    "Add",
                    "Mul",
                    "Resize",
                    "Conv",
                    "MatMul",
                    "MaxPool",
                ]:
                    quantized_blocks.append(couple_idx)
                    break

        return quantized_blocks

    def __build_support_graph(self, model: onnx.ModelProto) -> nx.DiGraph:
        support_graph = nx.DiGraph()

        model_input_names = [input.name for input in model.graph.input]
        model_output_names = [output.name for output in model.graph.output]

        support_graph.add_node("InputNode", node=None)
        support_graph.add_node("OutputNode", node=None)

        for node in model.graph.node:
            support_graph.add_node(node.name, node=node)

        ## Initializing queue to explore onnx model
        queue = []
        for node in model.graph.node:
            for model_input in model_input_names:
                if model_input in node.input:
                    queue.append(node)

        explored = set()
        while queue:
            curr_node: onnx.NodeProto = queue.pop(0)
            for out_name in curr_node.output:
                for next_node in model.graph.node:
                    if out_name in next_node.input:
                        if (curr_node.name, next_node.name) not in support_graph.edges:
                            support_graph.add_edge(
                                curr_node.name, next_node.name, tensors=set()
                            )
                        support_graph.edges[curr_node.name, next_node.name][
                            "tensors"
                        ].add(out_name)

                        if next_node.name not in explored:
                            queue.append(next_node)
            explored.add(curr_node.name)

        ## Adding connections from input node and to output node
        for node in model.graph.node:
            for in_name in node.input:
                if in_name in model_input_names:
                    if ("InputNode", node.name) not in support_graph.edges:
                        support_graph.add_edge("InputNode", node.name, tensors=set())
                    support_graph.edges["InputNode", node.name]["tensors"].add(in_name)

            for out_name in node.output:
                if out_name in model_output_names:
                    if (node.name, "OutputNode") not in support_graph.edges:
                        support_graph.add_edge(node.name, "OutputNode", tensors=set())
                    support_graph.edges[node.name, "OutputNode"]["tensors"].add(
                        out_name
                    )

        isolated = list(nx.isolates(support_graph))
        support_graph.remove_nodes_from(isolated)

        return support_graph

    def __build_mixed_model(
        self,
        original_model: onnx.ModelProto,
        quantized_model: onnx.ModelProto,
        blocks_num: int,
    ) -> onnx.ModelProto:

        original_model = onnx.shape_inference.infer_shapes(original_model)
        quantized_model = onnx.shape_inference.infer_shapes(quantized_model)

        q_tensor_names = [ten.name for ten in quantized_model.graph.value_info]

        original_model_graph: nx.DiGraph = self.__build_support_graph(original_model)
        original_topo_sort = list(nx.topological_sort(original_model_graph))

        indexes = np.arange(1, len(original_topo_sort) - 1)
        parts = np.array_split(indexes, blocks_num)

        valid_quantized_blocks: list[int] = self.__find_valid_quantized_blocks(
            original_model_graph, original_topo_sort, parts
        )

        nq_extractor = onnx.utils.Extractor(original_model)
        q_extractor = onnx.utils.Extractor(quantized_model)

        nq_extracted_models = []
        q_extracted_models = []

        nodes_list = []
        cond_inp_list = []

        for part_idx, part in enumerate(parts):

            first_node_idx = part[0]
            last_node_idx = part[-1]
            print(first_node_idx, last_node_idx)
            block_nodes = original_topo_sort[first_node_idx : last_node_idx + 1]

            input_names = set()
            output_names = set()

            exit_nodes = set()

            for node in block_nodes:
                for in_edge in original_model_graph.in_edges(node):
                    input_node = in_edge[0]
                    input_node_idx = original_topo_sort.index(input_node)
                    if input_node_idx < first_node_idx:
                        input_names.update(
                            original_model_graph.edges[in_edge].get("tensors")
                        )

                for out_edge in original_model_graph.out_edges(node):
                    output_node = out_edge[1]
                    output_node_idx = original_topo_sort.index(output_node)
                    if output_node_idx > last_node_idx:
                        output_names.update(
                            original_model_graph.edges[out_edge].get("tensors")
                        )
                        exit_nodes.add(out_edge[0])

            nq_extracted = nq_extractor.extract_model(
                input_names=input_names, output_names=output_names
            )
            nq_extracted_models.append(nq_extracted)

            if part_idx not in valid_quantized_blocks:
                ## If the block does not have valid quantized nodes
                ## Then the quantized version is the same as not quantized
                q_extracted = nq_extractor.extract_model(
                    input_names=input_names, output_names=output_names
                )
            else:
                q_output_names = set(output_names)
                for exit_node in exit_nodes:
                    if original_model_graph.nodes[exit_node].get("node").op_type in [
                        "Add",
                        "Mul",
                        "Resize",
                        "Conv",
                        "MatMul",
                        "MaxPool",
                    ]:
                        out_edges = original_model_graph.out_edges(exit_node)
                        for out_edge in out_edges:
                            for out_name in original_model_graph.edges[out_edge].get(
                                "tensors"
                            ):
                                q_name = out_name + "_DequantizeLinear_Output"
                                if q_name in q_tensor_names:
                                    q_output_names = q_output_names - set([out_name])
                                    q_output_names.add(q_name)

                q_extracted = q_extractor.extract_model(
                    input_names=input_names, output_names=q_output_names
                )

            ## Alligning tensor names between nq and q extracted models
            gs_q_extracted = gs.import_onnx(q_extracted)
            gs_q_extracted_tensors = gs_q_extracted.tensors()
            for tensor in gs_q_extracted_tensors.values():
                tensor.name = tensor.name + "_q_extracted"
            for out_tens in gs_q_extracted.outputs:
                for nq_out_tens_name in output_names:
                    if out_tens.name.startswith(nq_out_tens_name):
                        out_tens.name = nq_out_tens_name
            for inp_tens in gs_q_extracted.inputs:
                for nq_inp_tens_name in input_names:
                    if inp_tens.name.startswith(nq_inp_tens_name):
                        inp_tens.name = nq_inp_tens_name
            q_extracted = gs.export_onnx(gs_q_extracted)

            # onnx.save_model(q_extracted, f"q_extracted_{part_idx}.onnx")

            q_extracted_models.append(q_extracted)

            block_if_node, block_cond_inp = self.__build_if_node(
                nq_extracted, q_extracted, part_idx
            )

            nodes_list.append(block_if_node)
            cond_inp_list.append(block_cond_inp)

        final_input = []
        for elem in original_model.graph.input:
            final_input.append(elem)
        final_input += cond_inp_list

        graph = onnx.helper.make_graph(
            nodes_list,
            "conditional_graph",
            final_input,
            original_model.graph.output,
            initializer=quantized_model.graph.initializer,
        )

        mixed_model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 20)]
        )
        onnx.checker.check_model(mixed_model, full_check=True)

        mixed_model = onnxslim.slim(mixed_model)
        mixed_model.ir_version = 11
        # onnx.save_model(mixed_model, "mixed_model.onnx")

        return mixed_model

    def __build_if_node(
        self,
        not_quant_block: onnx.ModelProto,
        quant_block: onnx.ModelProto,
        block_idx: int,
    ):
        if_block_inputs = []  # [tens for tens in not_quant_block.graph.input]
        if_block_out_names = [tens.name for tens in not_quant_block.graph.output]
        if_block_out_names.sort()

        then_out_value_info = [elem for elem in quant_block.graph.output]
        then_out_value_info.sort(key=lambda x: x.name)

        then_graph = onnx.helper.make_graph(
            nodes=quant_block.graph.node,
            name=f"Branch_Block_{block_idx}_is_quant",
            inputs=if_block_inputs,
            outputs=then_out_value_info,
            initializer=quant_block.graph.initializer,
            value_info=quant_block.graph.value_info,
        )

        else_out_value_info = [elem for elem in not_quant_block.graph.output]
        else_out_value_info.sort(key=lambda x: x.name)

        else_graph = onnx.helper.make_graph(
            nodes=not_quant_block.graph.node,
            name=f"Branch_Block_{block_idx}_is_not_quant",
            inputs=if_block_inputs,
            outputs=else_out_value_info,
            initializer=not_quant_block.graph.initializer,
            value_info=not_quant_block.graph.value_info,
        )

        cond_name = self.block_format_string.format(block_idx=block_idx)

        cond_input = onnx.helper.make_tensor_value_info(cond_name, TensorProto.BOOL, [])
        if_node = onnx.helper.make_node(
            "If",
            inputs=[cond_name],
            outputs=if_block_out_names,
            name=f"If_block_{block_idx}_is_quant",
            then_branch=then_graph,
            else_branch=else_graph,
        )

        # onnx.checker.check_node(if_node)

        return if_node, cond_input
