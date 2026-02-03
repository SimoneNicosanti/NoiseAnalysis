import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from modelopt.onnx.quantization import quantize

from analyzer.AccuracyComputer import AccuracyFunction
from analyzer.NoiseFunction import NoiseFunction
from DatasetWrapper import DatasetWrapper
from model_preprocess.PPP import PPP
from quantization.ConditionalModelBuilder import ConditionalModelBuilder


class NoiseAnalyzer:
    def __init__(
        self,
        model_path: str,
        dataset_wrapper: DatasetWrapper,
        calib_size: int,
        eval_size: int,
        providers: list[str],
        batch_size: int,
        ppp: PPP,
        blocks_num: int,
    ):
        self.ppp: PPP = ppp
        self.batch_size = batch_size

        calib_dataset_wrapper: DatasetWrapper = dataset_wrapper.get_dataset_cut(
            0, calib_size
        )
        pre_processed_calib_data = self.ppp.preprocess(calib_dataset_wrapper.data)

        with tempfile.NamedTemporaryFile() as temp_file:

            quantize(
                onnx_path=model_path,
                output_path=temp_file.name,
                quantize_mode="int8",  # fp8, int8, int4 etc.
                calibration_data=pre_processed_calib_data,
                calibration_method="max",  # max, entropy, awq_clip, rtn_dq etc.
                high_precision_dtype="fp32",
                mha_accumulation_dtype="fp32",
                log_level="INFO",
                calibration_eps=["cuda:0"],
            )

            model_builder: ConditionalModelBuilder = ConditionalModelBuilder(model)
            conditional_model, self.execution_wrapper = (
                model_builder.build_conditional_model(
                    model_path, temp_file.name, blocks_num
                )
            )

        self.providers = providers
        self.sess = self.__build_inference_session(conditional_model)

        self.eval_dataset_wrapper: DatasetWrapper = dataset_wrapper.get_dataset_cut(
            calib_size, calib_size + eval_size
        )
        self.pre_processed_eval_data = self.ppp.preprocess(
            self.eval_dataset_wrapper.data
        )
        self.pre_processed_ort_eval_data = self.__prepare_eval_ort_values(
            self.pre_processed_eval_data, self.batch_size, eval_size
        )

        self.blocks_num = blocks_num
        # curr_quant_list = np.zeros(self.blocks_num, dtype=bool)

        self.output_names = [
            output_tensor.name for output_tensor in onnx.load(model_path).graph.output
        ]
        # self.original_raw_results = self.execution_wrapper.run(
        #     self.pre_processed_ort_eval_data,
        #     np.zeros(self.blocks_num, dtype=bool),
        #     self.output_names,
        # )
        self.original_raw_results = self._compute_model_results(
            self.pre_processed_ort_eval_data, []
        )

        pass

    def __prepare_eval_ort_values(
        self, eval_data_dict: dict[str, np.ndarray], batch_size: int, eval_size: int
    ) -> list[dict[str, ort.OrtValue]]:
        batches = []
        for cut_idx in range(0, eval_size, batch_size):
            batch_dict = {}
            for key in eval_data_dict.keys():
                ort_batch = ort.OrtValue.ortvalue_from_numpy(
                    eval_data_dict[key][cut_idx : cut_idx + batch_size], "cuda", 0
                )
                batch_dict[key] = ort_batch
            batches.append(batch_dict)
        return batches

    def __build_inference_session(self, model: onnx.ModelProto):
        # onnx.save_model(model, "conditional_model.onnx")
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = (
            1  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
        )
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        sess = ort.InferenceSession(
            model.SerializeToString(),
            providers=self.providers,
            sess_options=sess_options,
        )

        return sess

    def _compute_model_results(
        self,
        batches: list[dict[str, ort.OrtValue]],
        quant_list: list[int],
    ) -> dict[str, np.ndarray]:

        output_names = [out.name for out in self.sess.get_outputs()]
        model_results = {out_name: [] for out_name in output_names}

        input_dict = {}
        for block_idx in range(self.blocks_num):
            input_name = ConditionalModelBuilder.block_format_string.format(
                block_idx=block_idx
            )
            input_dict[input_name] = ort.OrtValue.ortvalue_from_numpy(
                np.array(block_idx in quant_list, dtype=np.bool_), device_type="cpu"
            )

        for batch in batches:
            input_dict.update(batch)
            result_list = self.sess.run_with_ort_values(output_names, input_dict)
            for out_idx, output_name in enumerate(model_results.keys()):
                model_results[output_name].append(result_list[out_idx].numpy())

        for out_name in model_results.keys():
            model_results[out_name] = np.concatenate(model_results[out_name], axis=0)

        return model_results

    def compute_quantized_model_results(
        self,
        curr_quant_blocks: list[int],
    ) -> dict[str, np.ndarray]:

        # quantized_results = self.execution_wrapper.run(
        #     self.pre_processed_ort_eval_data,
        #     np.array(
        #         [
        #             True if i in curr_quant_blocks else False
        #             for i in range(self.blocks_num)
        #         ]
        #     ),
        #     self.output_names,
        # )

        quantized_results = self._compute_model_results(
            self.pre_processed_ort_eval_data, curr_quant_blocks
        )

        return quantized_results

    def compute_noise_on_results(
        self, quant_raw_results: dict[str, np.ndarray], noise_function: NoiseFunction
    ) -> dict[str, float]:
        return noise_function.__call__(self.original_raw_results, quant_raw_results)

    def compute_accuracy_on_results(
        self, raw_results: dict[str, np.ndarray], accuracy_computer: AccuracyFunction
    ) -> dict[str, float]:
        post_proc_results = self.ppp.postprocess(
            self.eval_dataset_wrapper.data,
            raw_results,
            score_thr=1e-2,
            iou_thr=None,
        )
        return accuracy_computer.__call__(
            post_proc_results, self.eval_dataset_wrapper.ground_truth
        )
