import argparse
import gzip
import io
import os
import pickle
import subprocess
import zipfile

import numpy as np
from model_preprocess.PPP import PPP
from model_preprocess.YoloPPP import YoloPPP
from PIL import Image

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/ultralytics/coco128"
RAW_DATASET_PATH = "../raw/coco128.zip"
INTERNAL_PATH = "coco128/images/train2017"
PP_DATASET_PATH = "../preprocessed/"


def download():
    os.makedirs(os.path.dirname(RAW_DATASET_PATH), exist_ok=True)

    command = f"curl -L -o {RAW_DATASET_PATH} {DATASET_URL}"
    # Comando shell come lista di argomenti
    # trunk-ignore(bandit/B603)
    result = subprocess.run(command.split(" "), capture_output=True, text=True)

    # output e codice di ritorno
    print(result.stdout)
    print(result.stderr)

    return result.returncode


def preprocess_for_model(ppp: PPP, input_names: list[str]):
    pp_dict = {input_name: [] for input_name in input_names}

    with zipfile.ZipFile(RAW_DATASET_PATH, "r") as zip_ref:
        # Lista tutti i file nella directory interna
        files_in_dir = [f for f in zip_ref.namelist() if f.startswith(INTERNAL_PATH)]

        total_pp = 0
        total_skipped = 0
        for file in files_in_dir:
            # Apri il file dallo ZIP come byte stream
            with zip_ref.open(file) as f:
                img_data = f.read()
                img = Image.open(io.BytesIO(img_data))
                img_array = np.asarray(img)[..., :3]

                if len(img_array.shape) != 3:
                    # print("Skipping file: ", file)
                    total_skipped += 1
                    continue
                else:
                    # print("Processing file: ", file)
                    total_pp += 1
                    pass
                pp_dict_elem = ppp.preprocess(img_array)

                for key in pp_dict_elem:
                    pp_dict[key].append(pp_dict_elem[key])

    print("Total files processed: ", total_pp)
    print("Total files skipped: ", total_skipped)
    return pp_dict


def preprocess(model_name: str):
    os.makedirs(os.path.dirname(PP_DATASET_PATH), exist_ok=True)

    ppp: PPP = None
    input_names = []
    match model_name:
        case "yolo11":
            ppp = YoloPPP(640, 640)
            input_names = ["images"]
            pass
        case _:
            print("No preprocessing class for this model")

    pp_dict = preprocess_for_model(ppp, input_names)
    for key in pp_dict:
        pp_dict[key] = np.concatenate(pp_dict[key], axis=0)
        print(pp_dict[key].shape)

    pp_file_path = os.path.join(PP_DATASET_PATH, model_name, "coco128.pkl.gz")
    with gzip.open(pp_file_path, "wb") as f:
        pickle.dump(pp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--preprocess", type=str, default=None, choices=["yolo11"])

    args = parser.parse_args()

    down_result = 0
    if args.download:
        down_result = download()
    if args.preprocess is not None and down_result == 0:
        preprocess(args.preprocess)


if __name__ == "__main__":
    main()
