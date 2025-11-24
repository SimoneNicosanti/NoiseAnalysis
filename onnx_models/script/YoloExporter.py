## This script needs ultralytics dependency

from ultralytics import YOLO
import os
import shutil

DESTINATION_PATH = "../yolo11"

model_sizes = ["n", "s", "m", "l", "x"]
model_variants = [""] # "-seg"

os.makedirs(DESTINATION_PATH, exist_ok=True)

for model_size in model_sizes:
    for model_variant in model_variants:
        yolo_model = YOLO(f"yolo11{model_size}{model_variant}")
        yolo_model.export(format="onnx")
        
        os.remove(f"yolo11{model_size}{model_variant}.pt")
        if model_variant == "" :
            changed_model_variant = "-det"
        else :
            changed_model_variant = model_variant
        shutil.move(f"./yolo11{model_size}{model_variant}.onnx", f"{DESTINATION_PATH}/yolo11{model_size}{changed_model_variant}.onnx")
        