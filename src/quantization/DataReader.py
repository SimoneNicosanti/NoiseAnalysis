import numpy as np
from onnxruntime.quantization import CalibrationDataReader


class DataReader(CalibrationDataReader):
    def __init__(
        self,
        calib_dict: dict[str, list[np.ndarray]],
        batch_size: int = 1,
    ):

        self.curr_elem = 0
        self.batch_size = batch_size
        self.calib_dict = calib_dict

        self.input_set_size = list(self.calib_dict.values())[0].shape[0]

        pass

    def get_next(self):

        if self.curr_elem >= self.input_set_size:
            return None

        if self.curr_elem + self.batch_size > self.input_set_size:
            curr_batch_size = self.input_set_size - self.curr_elem
        else:
            curr_batch_size = self.batch_size

        input_dict = {}
        for input_name in self.calib_dict.keys():
            input_batch = self.calib_dict[input_name][
                self.curr_elem : self.curr_elem + curr_batch_size
            ]
            input_dict[input_name] = input_batch

        self.curr_elem += curr_batch_size
        return input_dict
