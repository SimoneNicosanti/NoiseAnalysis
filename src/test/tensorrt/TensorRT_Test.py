import argparse
import gzip
import os
import pickle
import time

import argcomplete
import numpy as np
import onnxruntime
import onnxruntime as ort
from modelopt.onnx.quantization import quantize

MODELS_BASE_PATH = "/workspaces/NoiseAnalysis/onnx_models"
DATASETS_BASE_PATH = "/workspaces/NoiseAnalysis/datasets/preprocessed"

OUTPUT_BASE_PATH = "/workspaces/NoiseAnalysis/results/built_dataset"


def read_dataset_dict(family: str, dataset_name: str):
    dataset_path = DATASETS_BASE_PATH + "/" + family + "/" + dataset_name + ".pkl.gz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found")

    dataset_dict = {}
    with gzip.open(dataset_path, "rb") as f:
        dataset_dict = pickle.load(f)

    return dataset_dict


def quantize_model(model_name: str):
    dataset_dict = read_dataset_dict("yolo11", "coco128")

    quantize(
        onnx_path=MODELS_BASE_PATH + f"/yolo11/{model_name}.onnx",
        quantize_mode="int8",  # fp8, int8, int4 etc.
        calibration_data=dataset_dict,
        calibration_method="max",  # max, entropy, awq_clip, rtn_dq etc.
        output_path=f"./{model_name}_quant.onnx",
        high_precision_dtype="fp32",
        mha_accumulation_dtype="fp32",
        log_level="INFO",
    )


def run_inference(model_name: str, batch_size: int, provider: str):
    sess_options = ort.SessionOptions()

    input = np.zeros((batch_size, 3, 640, 640), dtype=np.float32)

    warmup_runs = 5
    timed_runs = 25

    if provider == "TRT":
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_fp16_enable": True,
            # Enable explicit shape profiles
            "trt_profile_min_shapes": "images:1x3x640x640",
            "trt_profile_opt_shapes": "images:8x3x640x640",
            "trt_profile_max_shapes": "images:16x3x640x640",
        }
        providers = [("TensorrtExecutionProvider", trt_options)]
    else:
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        sess_options.optimized_model_filepath = f"{model_name}_optimized.onnx"
        providers = ["CUDAExecutionProvider"]

    sess = ort.InferenceSession(
        f"{model_name}.onnx",
        providers=providers,
        sess_options=sess_options,
    )

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    ## Warm up
    for _ in range(warmup_runs):
        sess.run([output_name], {input_name: input})

    ## Timed run
    times = np.zeros(timed_runs)

    for i in range(timed_runs):
        start = time.perf_counter_ns()
        sess.run([output_name], {input_name: input})
        end = time.perf_counter_ns()
        time_ns = end - start
        times[i] = time_ns
    print(
        "Model: {}, Provider: {}, Batch size: {}, Avg inference time: {} s".format(
            model_name, provider, batch_size, np.mean(times) / 1e9
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="TensorRT / CUDA Quantization and Inference Test"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--case",
        type=str,
        choices=["quantize", "inference"],
        required=True,
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["TRT", "CUDA"],
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    case = args.case
    if case == "quantize":
        quantize_model(args.model_name)
    elif case == "inference":
        if not args.provider:
            raise ValueError("Provider must be specified for inference case")
        run_inference(args.model_name, args.batch_size, args.provider)
    pass


def hook_configure_ort():
    import importhook

    @importhook.on_import("modelopt.onnx.quantization.ort_utils")
    def on_ort_utils_import(ort_utils):
        from modelopt.onnx.logging_config import logger

        new_ort_utils = importhook.copy_module(ort_utils)

        ## Patching configure_ort function

        original_configure_ort = new_ort_utils.configure_ort

        def patched_configure_ort(*args, **kwargs):
            logger.info("Running hooked version of configure_ort")

            fixed_args = []
            for a in args:
                fixed_args.append(a)
            fixed_args[0] = []
            fixed_args = tuple(fixed_args)

            op_types_list = kwargs.get("op_types", [])
            op_types_list.clear()
            return original_configure_ort(*fixed_args, **kwargs)

        new_ort_utils.configure_ort = patched_configure_ort

        import sys

        for m in sys.modules.values():
            try:
                for name, val in vars(m).items():
                    if val is original_configure_ort:
                        setattr(m, name, patched_configure_ort)
            except Exception:
                pass

        print('"ort_utils" module has been imported and hooked.')
        return new_ort_utils

    pass


if __name__ == "__main__":

    hook_configure_ort()

    main()
