import argparse
import gzip
import os
import pickle
import time

import numpy as np
import onnx_tool
from onnxruntime.quantization.quantize import QuantConfig

from analyzer.NoiseAnalyzer import NoiseAnalyzer
from analyzer.NoiseFunction import InfNorm, L1Norm, L1NormAvg, L2Norm, L2NormAvg
from dataset_builder.DatasetBuilder import DatasetBuilder

MODELS_BASE_PATH = "../onnx_models"
DATASETS_BASE_PATH = "../datasets/preprocessed"


def get_nodes_to_quantize_and_types(
    model_path: str, input_sizes: list[tuple], tot_nodes_to_quantize: int
) -> tuple[list[str], list[str]]:
    m = onnx_tool.Model(model_path)
    input_names = [input_name for input_name in m.graph.input if len(input_name) > 0]

    input_dict = {}
    for input_name in input_names:
        input_dict[input_name] = np.ones(input_sizes[0], dtype=np.float32)

    m.graph.shape_infer(input_dict)
    m.graph.profile()

    nodes_info = []
    for key in m.graph.nodemap.keys():
        node = m.graph.nodemap[key]
        node_tuple = (key, node.macs[0], node.op_type)
        nodes_info.append(node_tuple)

    top_nodes = sorted(nodes_info, key=lambda x: x[1], reverse=True)[
        :tot_nodes_to_quantize
    ]
    top_nodes_names = [node[0] for node in top_nodes]
    top_nodes_types = list(set([node[2] for node in top_nodes]))

    return top_nodes_names, top_nodes_types


def get_input_sizes(model_family: str):

    match model_family:
        case "yolo11":
            return [(1, 3, 640, 640)]
        case _:
            raise ValueError("No input sizes for this model")

    return []


def read_dataset_dict(family: str, dataset_name: str):
    dataset_path = DATASETS_BASE_PATH + "/" + family + "/" + dataset_name + ".pkl.gz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found")

    dataset_dict = {}
    with gzip.open(dataset_path, "rb") as f:
        # trunk-ignore(bandit/B301)
        dataset_dict = pickle.load(f)

    return dataset_dict


def main():
    parser = argparse.ArgumentParser()

    ## Model Information
    parser.add_argument("--family", type=str, required=True, choices=["yolo11"])
    parser.add_argument("--variant", type=str, required=True)

    ## Inference Information
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")

    ## Calibration Information
    parser.add_argument("--dataset", type=str, required=True, choices=["coco128"])
    # parser.add_argument("--calib-size", type=int, required=True)
    # parser.add_argument("--eval-size", type=int, required=True)

    # ## Quantization Information
    # ## TODO Here we can add all quantization information and variants if and when needed
    parser.add_argument("--layers-num", type=int, required=True)
    # parser.add_argument("--layers-type", type=str)

    # ## Predictor Training and Evaluation Information
    # parser.add_argument("--train-size", type=int, required=True)
    # parser.add_argument("--test-size", type=int, required=True)

    args = parser.parse_args()
    layers_num = args.layers_num
    batch = args.batch

    model_name = args.family + args.variant

    model_path = MODELS_BASE_PATH + "/" + args.family + "/" + model_name + ".onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found")

    input_sizes = get_input_sizes(args.family)
    nodes_names, nodes_types = get_nodes_to_quantize_and_types(
        model_path, input_sizes, tot_nodes_to_quantize=layers_num
    )

    dataset_dict = read_dataset_dict(args.family, args.dataset)

    providers = ["CPUExecutionProvider"]
    if args.gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    start = time.perf_counter_ns()
    quant_config = QuantConfig(
        op_types_to_quantize=nodes_types, nodes_to_quantize=nodes_names
    )
    noise_analyzer = NoiseAnalyzer(
        model_path,
        dataset_dict,
        calib_size=100,
        eval_size=20,
        providers=providers,
        batch_size=batch,
        quant_config=quant_config,
    )
    end = time.perf_counter_ns()
    print(f"Calibration Time >> {(end - start) / 1e9} s")

    res = noise_analyzer.compute_avg_noise_on_nodes(
        nodes_names, [L2Norm(), L1NormAvg()]
    )
    print(res)

    res = noise_analyzer.compute_avg_noise_on_nodes(
        nodes_names[:3], [L2Norm(), L1NormAvg()]
    )
    print(res)
    # quant_config = QuantConfig(
    #     nodes_to_quantize=nodes_names,
    #     op_types_to_quantize=nodes_types,
    # )

    # start = time.perf_counter_ns()
    # # avg_noise = noise_analyzer.compute_avg_noise(
    # #     quantization_config=quant_config,
    # #     extra_options={},
    # #     noise_functions=[NoiseFunction.L2NormAvg(), NoiseFunction.L1NormAvg()],
    # # )
    # dataset_builder = DatasetBuilder()
    # dataset_builder.build_dataset(noise_analyzer, nodes_names, nodes_types, 20, 20)
    # end = time.perf_counter_ns()

    # print("Time: ", (end - start) / 1e9)
    # # print("Average noise: ", avg_noise)
    pass


if __name__ == "__main__":
    main()
