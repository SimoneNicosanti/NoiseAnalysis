# import numpy as np
# import onnx
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import quantize_static

from quantization.DataReader import DataReader

reader = DataReader(
    {
        "images": np.zeros((1, 3, 640, 640), dtype=np.float32),
    },
    batch_size=1,
)


target_node_name = "/model.1/conv/Conv"

onnx.utils.extract_model(
    "../onnx_models/yolo11/yolo11n-det.onnx",
    "extracted_model.onnx",
    input_names=["/model.0/act/Mul_output_0"],
    output_names=["/model.1/conv/Conv_output_0"],
)
extracted_orig_model = onnx.load_model("extracted_model.onnx")

# quantize_static(
#     "../onnx_models/yolo11/yolo11n-det.onnx",
#     "quantized_model.onnx",
#     reader,
#     nodes_to_quantize=[target_node_name],
# )

onnx.checker.check_model(onnx.load_model("quantized_model.onnx"))

onnx.utils.extract_model(
    "./quantized_model.onnx",
    "extracted_model_2.onnx",
    input_names=["/model.0/act/Mul_output_0"],
    output_names=["/model.1/conv/Conv_output_0_DequantizeLinear_Output"],
    check_model=False,
)
extracted_quant_model = onnx.load_model("extracted_model_2.onnx")

model_input_names = [tens.name for tens in extracted_quant_model.graph.input]
model_output_names = [tens.name for tens in extracted_quant_model.graph.output]

for tens in extracted_quant_model.graph.initializer:
    ## TODO
    if tens.name not in model_input_names and tens.name not in model_output_names:
        tens.name = tens.name + ".branch"

for tens in extracted_quant_model.graph.value_info:
    ## TODO
    if tens.name not in model_input_names and tens.name not in model_output_names:
        tens.name = tens.name + ".branch"

for node in extracted_quant_model.graph.node:
    ## TODO
    for i, out_name in enumerate(node.output):
        if out_name not in model_input_names and out_name not in model_output_names:
            node.output[i] = out_name + ".branch"
    for i, in_name in enumerate(node.input):
        if in_name not in model_input_names and in_name not in model_output_names:
            node.input[i] = in_name + ".branch"
    pass

onnx.checker.check_model(extracted_quant_model)
onnx.save_model(extracted_quant_model, "quantized_model.onnx")


if_graph = helper.make_graph(
    nodes=extracted_quant_model.graph.node,
    name=f"If_{target_node_name}_is_quant",
    inputs=[],  # <-- tensore dichiarato come input
    outputs=extracted_quant_model.graph.output,
    initializer=extracted_quant_model.graph.initializer,
    value_info=extracted_quant_model.graph.value_info,
)

else_graph = helper.make_graph(
    nodes=extracted_orig_model.graph.node,
    name=f"If_{target_node_name}_is_not_quant",
    inputs=[],  # <-- tensore dichiarato come input
    outputs=extracted_orig_model.graph.output,
    initializer=extracted_orig_model.graph.initializer,
    value_info=extracted_orig_model.graph.value_info,
)

cond_input = onnx.helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
if_node = helper.make_node(
    "If",
    inputs=["cond"],
    outputs=["/model.1/conv/Conv_output_0"],
    name=f"If_{target_node_name}",
    then_branch=if_graph,
    else_branch=else_graph,
)

original_model = onnx.load_model("../onnx_models/yolo11/yolo11n-det.onnx")

i = 0
target_node = None
for node in original_model.graph.node:
    if node.name == target_node_name:
        target_node = node
        break
    i += 1

original_model.graph.node.remove(target_node)
original_model.graph.node.insert(i, if_node)

original_model.graph.input.append(cond_input)

new_model = onnx.helper.make_model(
    original_model.graph,
    producer_name="dummy_conv_if",
    opset_imports=[onnx.helper.make_operatorsetid("", 18)],
    ir_version=11,
)

for node in new_model.graph.node:
    for i, out_name in enumerate(node.output):
        if out_name == "model.1.conv.bias":
            node.output[i] = "model.1.conv.bias.branch"  # any unique name


onnx.checker.check_model(new_model)
onnx.save_model(new_model, "dummy_conv_with_if.onnx")

import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "dummy_conv_with_if.onnx", providers=["CUDAExecutionProvider"]
)
out = sess.run(
    None,
    {
        "cond": np.array(False, dtype=np.bool_),
        "images": np.zeros((1, 3, 640, 640), dtype=np.float32),
    },
)
print(out)

out = sess.run(
    None,
    {
        "cond": np.array(True, dtype=np.bool_),
        "images": np.zeros((1, 3, 640, 640), dtype=np.float32),
    },
)
print(out)
