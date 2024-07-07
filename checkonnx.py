import onnx

# 載入ONNX模型
onnx_model = onnx.load("simple_model.onnx")
onnx.checker.check_model(onnx_model)

for input in onnx_model.graph.input:
    print(input.name)
for output in onnx_model.graph.output:
    print(output.name)
