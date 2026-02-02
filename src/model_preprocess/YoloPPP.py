import cv2
import numpy as np

from model_preprocess.PPP import PPP


class YoloPPP(PPP):
    def __init__(
        self,
        mod_input_height: int,
        mod_input_width: int,
        mod_classes_dict: dict[int, str],
        trg_classes_dict: dict[int, str],
    ):
        self.mod_input_height = mod_input_height
        self.mod_input_width = mod_input_width

        self.class_mapping = self.__build_classes_mapping(
            mod_classes_dict, trg_classes_dict
        )

    def __build_classes_mapping(
        self, mod_classes: dict[int, str], trg_classes: dict[int, str]
    ):
        classes_mapping = {}
        num_not_found = 0
        for mod_class_id, mod_class_lab in mod_classes.items():
            found = False
            for trg_class_id, trg_class_lab in trg_classes.items():
                if mod_class_lab == trg_class_lab:
                    classes_mapping[mod_class_id] = trg_class_id
                    found = True
                    break

            if not found:
                classes_mapping[mod_class_id] = len(trg_classes) + num_not_found
                num_not_found += 1

        return classes_mapping

    def compute_ratio(self, input_image: np.ndarray):
        shape = input_image.shape[:2]
        new_shape = (self.mod_input_height, self.mod_input_width)

        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        return ratio

    def compute_unpad_shape(self, input_image: np.ndarray, ratio: float):
        shape = input_image.shape[:2]
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))

        return new_unpad

    def compute_pad(self, input_image: np.ndarray, ratio: float, unpad_shape: tuple):
        new_shape = (self.mod_input_height, self.mod_input_width)
        pad_w, pad_h = (new_shape[1] - unpad_shape[0]) / 2, (
            new_shape[0] - unpad_shape[1]
        ) / 2  # wh padding

        return pad_w, pad_h
        pass

    def preprocess(self, original: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Batch pre-processing with letterbox, equivalent to Ultralytics behavior.

        Args:
            images: list of HWC BGR uint8 images (variable sizes)

        Returns:
            dict with:
                images: (B, 3, input_h, input_w) float32 in RGB [0,1]
                ratio:  (B, 2)
                pad:    (B, 2)
        """
        images = original["images"]
        batch_size = len(images)

        # Original shapes
        shapes = np.array(
            [img.shape[:2] for img in images], dtype=np.float32
        )  # (B, 2) -> (H, W)
        h0, w0 = shapes[:, 0], shapes[:, 1]

        # Target shape
        new_h, new_w = self.mod_input_height, self.mod_input_width

        # Compute scale ratio (YOLO-style: min ratio, keep aspect)
        r = np.minimum(new_h / h0, new_w / w0)
        # ratio = np.stack([r, r], axis=1)  # (B, 2)

        # Unpadded resized shape
        new_unpad_h = np.round(h0 * r)
        new_unpad_w = np.round(w0 * r)

        # Padding
        pad_h = (new_h - new_unpad_h) / 2
        pad_w = (new_w - new_unpad_w) / 2

        # Output tensor
        batch = np.full(
            (batch_size, new_h, new_w, 3),
            114,
            dtype=np.uint8,
        )

        # Resize + place (cannot be vectorized due to OpenCV)
        for i, img in enumerate(images):
            resized = cv2.resize(
                img,
                (int(new_unpad_w[i]), int(new_unpad_h[i])),
                interpolation=cv2.INTER_LINEAR,
            )

            top = int(round(pad_h[i] - 0.1))
            left = int(round(pad_w[i] - 0.1))

            batch[
                i, top : top + resized.shape[0], left : left + resized.shape[1], :
            ] = resized

        # Vectorized tensor transforms
        # HWC -> BCHW, BGR -> RGB, float32, normalize
        batch = batch.transpose(0, 3, 1, 2)[:, ::-1, :, :]
        batch = np.ascontiguousarray(batch, dtype=np.float32) / 255.0

        return {
            "images": batch,
        }

    def postprocess(
        self,
        original: dict[str, np.ndarray],
        outputs: dict[str, np.ndarray],
        **kwargs,
    ) -> dict[str, list[np.ndarray] | None]:

        original_images = original["images"]

        predictions = outputs["output0"]
        prototypes = outputs.get("output1", None)

        score_thr = kwargs.get("score_thr", 1e-3)
        iou_thr = kwargs.get("iou_thr", None)
        num_classes = len(self.class_mapping)

        B = predictions.shape[0]

        # ---- letterbox params ----
        shapes = np.array([img.shape[:2] for img in original_images], dtype=np.float32)
        h0, w0 = shapes[:, 0], shapes[:, 1]

        in_h, in_w = self.mod_input_height, self.mod_input_width
        r = np.minimum(in_h / h0, in_w / w0)
        ratio = r[:, None]

        new_h = np.round(h0 * r)
        new_w = np.round(w0 * r)

        pad_h = (in_h - new_h) / 2
        pad_w = (in_w - new_w) / 2

        # ---- transpose ----
        x = np.einsum("bcn->bnc", predictions)

        scores_all = np.max(x[..., 4 : 4 + num_classes], axis=-1)
        classes_all = np.argmax(x[..., 4 : 4 + num_classes], axis=-1)
        keep = scores_all > score_thr

        results_boxes = []
        results_scores = []
        results_labels = []
        results_masks = []
        results_segments = []

        for i in range(B):
            xi = x[i][keep[i]]
            if xi.size == 0:
                results_boxes.append(None)
                results_scores.append(None)
                results_labels.append(None)
                results_masks.append(None)
                results_segments.append(None)
                continue

            scores = scores_all[i][keep[i]]
            labels = classes_all[i][keep[i]]

            xi = np.c_[
                xi[:, :4],  # cxcywh
                scores,  # score
                labels,  # class
                xi[:, 4 + num_classes :],  # mask coeffs
            ]

            # ---- optional NMS ----
            if iou_thr is not None:
                idx = cv2.dnn.NMSBoxes(
                    xi[:, :4].tolist(),
                    xi[:, 4].tolist(),
                    score_thr,
                    iou_thr,
                )
                if len(idx) == 0:
                    results_boxes.append(None)
                    results_scores.append(None)
                    results_labels.append(None)
                    results_masks.append(None)
                    results_segments.append(None)
                    continue
                xi = xi[idx.flatten()]

            # ---- decode boxes ----
            xi[:, 0] -= xi[:, 2] / 2
            xi[:, 1] -= xi[:, 3] / 2
            xi[:, 2] += xi[:, 0]
            xi[:, 3] += xi[:, 1]

            xi[:, [0, 2]] -= pad_w[i]
            xi[:, [1, 3]] -= pad_h[i]
            xi[:, :4] /= ratio[i]

            h, w = original_images[i].shape[:2]
            xi[:, [0, 2]] = xi[:, [0, 2]].clip(0, w)
            xi[:, [1, 3]] = xi[:, [1, 3]].clip(0, h)

            boxes = xi[:, :4]
            scores = xi[:, 4]
            labels = xi[:, 5].astype(np.int32)

            results_boxes.append(boxes)
            results_scores.append(scores)
            results_labels.append(labels)

            # ---- masks ----
            if prototypes is not None:
                masks = self.__process_mask(
                    prototypes[i],
                    xi[:, 6:],
                    boxes,
                    original_images[i].shape,
                )
                segments = self.__masks2segments(masks)
            else:
                masks = None
                segments = None

            results_masks.append(masks)
            results_segments.append(segments)

        for label_array in results_labels:
            if label_array is None:
                continue
            for i in range(len(label_array)):
                label_array[i] = self.class_mapping[label_array[i]]

        return {
            "boxes": results_boxes,
            "scores": results_scores,
            "labels": results_labels,
            "masks": results_masks,
            "segments": results_segments,
        }

    @staticmethod
    def __masks2segments(masks):
        """
        Takes a list of masks(n,h,w) and returns a list of segments(n,xy), from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                0
            ]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments

    @staticmethod
    def __crop_mask(masks, boxes):
        """
        Takes a mask and a bounding box, and returns a mask that is cropped to the bounding box, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def __process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
        quality but is slower, from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = (
            np.matmul(masks_in, protos.reshape((c, -1)))
            .reshape((-1, mh, mw))
            .transpose(1, 2, 0)
        )  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.__scale_mask(
            masks, im0_shape
        )  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.__crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)

    @staticmethod
    def __scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(
                im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1]
            )  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (
                im1_shape[0] - im0_shape[0] * gain
            ) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(
            round(im1_shape[1] - pad[0] + 0.1)
        )
        if len(masks.shape) < 2:
            raise ValueError(
                f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
            )
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks
