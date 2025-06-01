import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession("resnet18.onnx")

# 查看输入输出名
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 准备输入数据（形状必须和导出时保持一致或符合 dynamic_axes）
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 运行推理
output = session.run([output_name], {input_name: input_data})

print("Output:", output[0].shape)