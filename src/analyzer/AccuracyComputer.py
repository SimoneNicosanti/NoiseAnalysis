from abc import ABC, abstractmethod

from supervision import Detections
from supervision.metrics.mean_average_precision import MeanAveragePrecision


class AccuracyFunction(ABC):

    @abstractmethod
    def __call__(self, results: dict[str, float], ground_truth: dict[str, float]):
        raise NotImplementedError


class YoloAccuracyFunction(AccuracyFunction):
    def __call__(self, post_proc_resuls: dict[str], ground_truth: dict[str]):
        metric = MeanAveragePrecision()

        for i in range(len(post_proc_resuls["boxes"])):
            curr_detection = Detections(
                post_proc_resuls["boxes"][i],
                class_id=post_proc_resuls["labels"][i],
                confidence=post_proc_resuls["scores"][i],
            )

            metric.update(curr_detection, ground_truth["detections"][i])

        metric_result = metric.compute()

        return {
            "map50": metric_result.map50,
            "map50_95": metric_result.map50_95,
            "map75": metric_result.map75,
        }
