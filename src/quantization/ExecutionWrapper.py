import numpy as np
import onnx
import onnxruntime as ort


class ExecutionWrapper:

    def __init__(
        self,
        nq_extracted_models: list[onnx.ModelProto],
        q_extracted_models: list[onnx.ModelProto],
    ):
        self.nq_extracted_models = {
            idx: nq_extracted_models
            for idx, nq_extracted_models in enumerate(nq_extracted_models)
        }
        self.q_extracted_models = {
            idx: q_extracted_models
            for idx, q_extracted_models in enumerate(q_extracted_models)
        }

        self.nq_models_sessions = {}
        self.q_models_sessions = {}

    def __init_model_sessions(self, idx: int, is_quantized: bool):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_fp16_enable": True,
            # Enable explicit shape profiles
            # "trt_profile_min_shapes": "images:1x3x640x640",
            # "trt_profile_opt_shapes": "images:8x3x640x640",
            # "trt_profile_max_shapes": "images:16x3x640x640",
        }
        providers = [("TensorrtExecutionProvider", trt_options)]

        if is_quantized:
            if idx not in self.q_models_sessions:
                q_model = self.q_extracted_models[idx]
                q_session = ort.InferenceSession(
                    q_model.SerializeToString(),
                    sess_options=sess_options,
                    providers=providers,
                )
                self.q_models_sessions[idx] = q_session
        else:
            if idx not in self.nq_models_sessions:
                nq_model = self.nq_extracted_models[idx]
                nq_session = ort.InferenceSession(
                    nq_model.SerializeToString(),
                    sess_options=sess_options,
                    providers=providers,
                )
                self.nq_models_sessions[idx] = nq_session

    def run(
        self,
        input_data: dict[str, np.ndarray],
        quantization_scheme: np.ndarray,
        output_list: list[str],
    ) -> dict[str, np.ndarray]:
        result_cache: dict[str, ort.OrtValue] = {}

        for key, value in input_data.items():
            result_cache[key] = ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)

        for idx in self.nq_extracted_models.keys():
            print("Running model ", idx)
            is_quantized = quantization_scheme[idx]
            self.__init_model_sessions(idx, is_quantized)

            session: ort.InferenceSession
            if is_quantized:
                session = self.q_models_sessions[idx]
            else:
                session = self.nq_models_sessions[idx]

            input_names = [input.name for input in session.get_inputs()]
            inputs_feed = {name: result_cache[name] for name in input_names}

            output_names = [output.name for output in session.get_outputs()]
            outputs = session.run_with_ort_values(output_names, inputs_feed)

            for name, output in zip(output_names, outputs, strict=False):
                result_cache[name] = output

        output = {output_name: result_cache[output_name] for output_name in output_list}
        return output
        pass
