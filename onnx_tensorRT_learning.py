import torch
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11,         # >=11 推荐
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)