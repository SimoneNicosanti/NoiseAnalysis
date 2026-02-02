import numpy as np

class DatasetWrapper :
    def __init__(self, ids : list, data : dict[str, np.ndarray], ground_truth : dict[str]) :
        self.ids = ids
        self.data = data
        self.ground_truth = ground_truth

    def get_dataset_cut(self, start : int, end : int) :
        cut_data = {}
        for input_name in self.data :
            cut_data[input_name] = self.data[input_name][start:end]
        cut_gt = {}
        for input_name in self.ground_truth :
            cut_gt[input_name] = self.ground_truth[input_name][start:end]
        
        return DatasetWrapper(self.ids[start:end], cut_data, cut_gt)

    
    def get_input_names(self) :
        return list(self.data.keys())