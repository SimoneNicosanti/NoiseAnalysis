import argparse
import os

import argcomplete
import pandas as pd
from supervision.dataset.core import DetectionDataset

from analyzer.AccuracyComputer import YoloAccuracyFunction
from analyzer.NoiseAnalyzer import NoiseAnalyzer
from dataset_builder.DatasetBuilder import DatasetBuilder
from DatasetWrapper import DatasetWrapper
from model_preprocess.YoloPPP import YoloPPP

MODELS_BASE_PATH = "../onnx_models"
DATASETS_BASE_PATH = "../datasets/preprocessed"
OUTPUT_BASE_PATH = "../results/built_dataset"
IMAGES_PATH = "../datasets/raw/coco-val/images/val"
LABELS_PATH = "../datasets/raw/coco-val/labels/val"
YAML_PATH = "../datasets/raw/coco-val/dataset.yaml"
MODEL_YAML_PATH = "../datasets/raw/coco-val/ultralytics.yaml"


def read_dataset_dict(family: str, dataset_name: str, calib_size: int, eval_size: int):

    if dataset_name == "coco128":
        dataset = DetectionDataset.from_yolo(
            images_directory_path=IMAGES_PATH,
            annotations_directory_path=LABELS_PATH,
            data_yaml_path=YAML_PATH,
        )
        ids = []
        data = []
        ground_truths = []
        for elem in dataset:
            ids.append(elem[0])
            data.append(elem[1])
            ground_truths.append(elem[2])
        dataset_wrapper = DatasetWrapper(
            ids, {"images": data}, {"detections": ground_truths}
        )

        classes_map = {
            idx: class_name for idx, class_name in enumerate(dataset.classes)
        }
    else:
        raise NotImplementedError
    return dataset_wrapper, classes_map


def save_built_dataset(
    dataset_dict: dict[str, pd.DataFrame],
    model_family: str,
    model_variant: str,
):
    output_path = OUTPUT_BASE_PATH + "/" + model_family + "/" + model_variant
    os.makedirs(output_path, exist_ok=True)

    for metric_name in dataset_dict:
        dataframe = dataset_dict[metric_name]
        dataframe.to_csv(f"./{output_path}/{metric_name}.csv", index=False)


def read_model_classes():
    import yaml

    with open(MODEL_YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    return data["names"]


def main(args):

    blocks_num = args.blocks_num
    batch = args.batch

    model_name = args.family + args.variant

    model_path = MODELS_BASE_PATH + "/" + args.family + "/" + model_name + ".onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found")

    mod_classes = read_model_classes()

    dataset_wrapper, trg_classes = read_dataset_dict(
        args.family, args.dataset, args.calib_size, args.eval_size
    )

    ppp = YoloPPP(640, 640, mod_classes, trg_classes)

    noise_analyzer = NoiseAnalyzer(
        model_path,
        dataset_wrapper,
        calib_size=args.calib_size,
        eval_size=args.eval_size,
        providers=["CUDAExecutionProvider"],
        batch_size=batch,
        blocks_num=blocks_num,
        ppp=ppp,
    )

    dataset_builder = DatasetBuilder(noise_analyzer)
    noise_dict = dataset_builder.build_dataset(
        blocks_num,
        args.dataset_size,
        accuracy_computer=YoloAccuracyFunction(),
        noise_functions=None,
    )

    save_built_dataset(noise_dict, args.family, args.variant)


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
            except Exception as e:
                raise e

        print('"ort_utils" module has been imported and hooked.')
        return new_ort_utils

    pass


if __name__ == "__main__":
    hook_configure_ort()
    parser = argparse.ArgumentParser()

    ## Model Information
    parser.add_argument("--family", type=str, required=True, choices=["yolo11"])
    parser.add_argument("--variant", type=str, required=True)

    ## Inference Information
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")

    ## Calibration Information
    parser.add_argument("--dataset", type=str, required=True, choices=["coco128"])
    parser.add_argument("--calib-size", type=int, required=True)
    parser.add_argument("--eval-size", type=int, required=True)

    ## Quantization Information
    parser.add_argument("--blocks-num", type=int, required=True)

    # ## Predictor Training and Evaluation Information
    parser.add_argument("--dataset-size", type=int, required=True)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    main(args)
