import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader


class DataReader(CalibrationDataReader):
    def __init__(self, model_path: str, calibration_set: np.ndarray):
        sess = ort.InferenceSession(model_path)
        input_info = sess.get_inputs()
        del sess

        self.input_names = [input.name for input in input_info]

        self.curr_elem = 0
        self.calibration_set = calibration_set

        pass

    def get_next(self):

        if self.curr_elem >= len(self.calibration_set):
            return None

        input_dict = {}
        for input_name in self.input_names:
            input_elem = self.calibration_set[self.curr_elem]
            input_dict[input_name] = input_elem

        self.curr_elem += 1
        return input_dict
