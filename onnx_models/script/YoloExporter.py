#!/opt/ultralytics-venv/bin/python
## This script needs ultralytics dependency
## As such, we launch it in the ultralytics venv created in the devcontainer


import argparse
import os
import shutil
import time

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

POSSIBLE_SIZES = ["all", "n", "s", "m", "l", "x"]
POSSIBLE_VARIANTS = ["all", "det", "seg"]



def export_yolo_model(family: str, variant: str, size: str, device: str):
    os.makedirs(f"../{family}", exist_ok=True)

    if variant == "all":
        model_variants = ["", "-seg"]
    else:
        model_variant = "" if variant == "det" else "-seg"
        model_variants = [model_variant]

    if size == "all":
        model_sizes = POSSIBLE_SIZES[1:]
    else:
        model_sizes = [size]

    if device == "cpu":
        target_device = "cpu"
    else:
        target_device = "0"

    for model_size in model_sizes:
        for model_variant in model_variants:
            yolo_model = YOLO(f"{family}{model_size}{model_variant}")
            yolo_model.export(format="onnx", device=target_device, dynamic=True)

            os.remove(f"{family}{model_size}{model_variant}.pt")
            if model_variant == "":
                changed_model_variant = "-det"
            else:
                changed_model_variant = model_variant

            shutil.move(
                f"./{family}{model_size}{model_variant}.onnx",
                f"../{family}/{family}{model_size}{changed_model_variant}.onnx",
            )


def run_yolo_model(family: str, variant: str, size: str, device: str, batch: int):
    pass
    model_path = f"../{family}/{family}{size}-{variant}.onnx"

    if device == "cuda":
        cuda_sess = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )
        start = time.perf_counter_ns()
        cuda_sess.run(
            None, {"images": np.zeros((batch, 3, 640, 640), dtype=np.float32)}
        )
        end = time.perf_counter_ns()
        print("CUDA Time: ", (end - start) / 1e9)

    cpu_sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    start = time.perf_counter_ns()
    cpu_sess.run(None, {"images": np.zeros((batch, 3, 640, 640), dtype=np.float32)})
    end = time.perf_counter_ns()
    print("CPU Time: ", (end - start) / 1e9)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["export", "run"])
    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda"])

    parser.add_argument("--family", type=str, default="yolo11")
    parser.add_argument("--size", type=str, required=True, choices=POSSIBLE_SIZES)
    parser.add_argument("--variant", type=str, required=True, choices=POSSIBLE_VARIANTS)

    parser.add_argument("--batch", type=int, required=False, default=1)

    args = parser.parse_args()

    family = args.family
    variant = args.variant
    size = args.size
    device = args.device

    if args.mode == "export":
        export_yolo_model(family, variant, size, device)
    if args.mode == "run" and size != "all" and variant != "all":
        run_yolo_model(family, variant, size, device, args.batch)


if __name__ == "__main__":
    ## IMPORTANT
    ## To get a dynamic shape on batch but not on width and heigh, I needed to change the library export
    ## The change has been made in ultralytics.engine.exporter in Exporter class at method export_onnx
    ## I changed from:
    #### dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}
    ## To:
    #### dynamic = {"images": {0: "batch"}}
    ## If you want to get the same effect, you will have to patch the export in some way as well
    main()
