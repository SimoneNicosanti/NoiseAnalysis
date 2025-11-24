import argparse
import os

from onnxruntime.quantization.quantize import QuantConfig

from analyzer import NoiseFunction
from analyzer.NoiseAnalyzer import NoiseAnalyzer

MODELS_BASE_PATH = "../onnx_models"
DATASETS_BASE_PATH = "../datasets/preprocessed"


def main():
    parser = argparse.ArgumentParser()

    ## Model Information
    parser.add_argument("--family", type=str, required=True, choices=["yolo11"])
    parser.add_argument("--variant", type=str, required=True)

    ## Calibration Information
    parser.add_argument("--dataset", type=str, required=True, choices=["coco128"])
    # parser.add_argument("--calib-size", type=int, required=True)
    # parser.add_argument("--eval-size", type=int, required=True)

    # ## Quantization Information
    # ## TODO Here we can add all quantization information and variants if and when needed
    # parser.add_argument("--layers-num", type=int, required=True)
    # parser.add_argument("--layers-type", type=str)

    # ## Predictor Training and Evaluation Information
    # parser.add_argument("--train-size", type=int, required=True)
    # parser.add_argument("--test-size", type=int, required=True)

    args = parser.parse_args()

    model_name = args.family + args.variant

    model_path = MODELS_BASE_PATH + "/" + args.family + "/" + model_name + ".onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found")

    dataset_path = DATASETS_BASE_PATH + "/" + args.family + "/" + args.dataset + ".npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found")

    noise_analyzer = NoiseAnalyzer(
        model_path, dataset_path, 100, 20, ["CPUExecutionProvider"]
    )
    quant_config = QuantConfig(op_types_to_quantize=["Conv"])

    avg_noise = noise_analyzer.compute_avg_noise(
        quant_config, {}, NoiseFunction.L2NormAvg()
    )
    print("Average noise: ", avg_noise)
    pass


if __name__ == "__main__":
    main()
