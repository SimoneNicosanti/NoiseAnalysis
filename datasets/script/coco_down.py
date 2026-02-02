import fiftyone as fo
import fiftyone.zoo as foz

EXPORT_DIR = "../raw/coco-val"
DATASET_SIZE = 500


def remap_labels(sample, model_names, name_to_model_id):
    detections = sample.detections.detections
    for det in detections:
        det.label = model_names[name_to_model_id[det.label]]
    sample.save()


def download(size):
    classes = foz.load_zoo_dataset_info("coco-2017").classes
    # print(classes)

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=classes,  # only this class
        max_samples=size,  # exactly one sample
        overwrite=True,
    )

    return dataset


def main():

    dataset = download(DATASET_SIZE)

    dataset.export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
    )


if __name__ == "__main__":
    main()
