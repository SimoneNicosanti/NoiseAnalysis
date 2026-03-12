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

        self.model_sessions = {}

    def __build_trt_profile_shapes(self, model: onnx.ModelProto):
        batch_dim_dict = {
            "trt_profile_min_shapes": 1,
            "trt_profile_opt_shapes": 3,
            "trt_profile_max_shapes": 5,
        }
        profile_shapes = {
            "trt_profile_min_shapes": "",
            "trt_profile_opt_shapes": "",
            "trt_profile_max_shapes": "",
        }
        for trt_profile_key in profile_shapes.keys():
            curr_batch_size = batch_dim_dict[trt_profile_key]
            for input in model.graph.input:
                input_name = input.name
                profile_shapes[trt_profile_key] += f"{input_name}:{curr_batch_size}"
                dimensions = input.type.tensor_type.shape.dim
                for dim in dimensions:
                    if dim.HasField("dim_param"):
                        ## This is the batch dimension
                        continue
                    else:
                        ## This is the tensor dimension
                        profile_shapes[trt_profile_key] += f"x{dim.dim_value}"
                profile_shapes[trt_profile_key] += ","

        return profile_shapes

    def __init_model_sessions(
        self, idx: int, is_quantized: bool
    ) -> ort.InferenceSession:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        trt_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_builder_optimization_level": "3",
            "trt_max_workspace_size": "536870912",  # 512MB
            # "trt_max_workspace_size": "2147483648",
        }
        providers = [("TensorrtExecutionProvider", trt_options)]

        if idx in self.model_sessions:
            if self.model_sessions[idx][0] == is_quantized:
                return self.model_sessions[idx][1]

            else:
                del self.model_sessions[idx]
                import gc

                gc.collect()

        curr_model = (
            self.q_extracted_models[idx]
            if is_quantized
            else self.nq_extracted_models[idx]
        )
        trt_profiles = self.__build_trt_profile_shapes(curr_model)
        trt_options.update(trt_profiles)

        session = ort.InferenceSession(
            curr_model.SerializeToString(),
            sess_options=sess_options,
            providers=["CUDAExecutionProvider"],
        )
        self.model_sessions[idx] = (is_quantized, session)
        return session

    def run(
        self,
        batches: list[dict[str, ort.OrtValue]],
        quantization_scheme: np.ndarray,
        output_list: list[str],
    ) -> dict[str, np.ndarray]:

        batches_outputs = []

        for batch in batches:
            result_cache: dict[str, ort.OrtValue] = {}
            result_cache.update(batch)

            for idx in self.nq_extracted_models.keys():
                is_quantized = quantization_scheme[idx]
                session: ort.InferenceSession = self.__init_model_sessions(
                    idx, is_quantized
                )

                input_names = [input.name for input in session.get_inputs()]
                inputs_feed = {name: result_cache[name] for name in input_names}

                output_names = [output.name for output in session.get_outputs()]
                outputs = session.run_with_ort_values(output_names, inputs_feed)

                for name, output in zip(output_names, outputs, strict=False):
                    result_cache[name] = output

            curr_output = {
                output_name: result_cache[output_name].numpy()
                for output_name in output_list
            }
            batches_outputs.append(curr_output)

        total_output = {
            output_name: np.concatenate(
                [output[output_name] for output in batches_outputs], axis=0
            )
            for output_name in output_list
        }

        return total_output
        pass
