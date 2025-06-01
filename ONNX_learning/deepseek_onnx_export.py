# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", 
                                          trust_remote_code=True
                                         )
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", 
                                             trust_remote_code=True,
                                             device_map="auto",          # 自动映射设备，若无GPU，全部加载CPU
                                             load_in_8bit=False          # 关闭8bit量化加载（如果支持）
                                        )

model.eval()  # 设置模型为评估模式
# 导出为 ONNX 格式
import torch
dummy_input = torch.randn(1, 3, 224, 224)  # 根据模型输入shape调整

torch.onnx.export(
    model,
    dummy_input,
    "deepseek_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("ONNX 导出完成")
