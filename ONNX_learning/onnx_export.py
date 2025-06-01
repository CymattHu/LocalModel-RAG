import torch
import onnx
import traceback
import gradio as gr
from io import BytesIO
import zipfile
import tarfile

def load_model_from_bytes(file_bytes, filename):
    ext = filename.lower()
    file_bytes.seek(0)
    data = file_bytes.read()

    if ext.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            return None, "❌ 未安装 safetensors 库，请先运行 `pip install safetensors`"

        try:
            # safetensors 加载为 state_dict
            state_dict = load_file(BytesIO(data))
            return state_dict, None
        except Exception as e:
            return None, f"❌ safetensors 文件加载失败: {e}"

    # 其他格式处理
    try:
        return torch.load(BytesIO(data), map_location="cpu"), None
    except Exception as e:
        # 尝试解压 zip
        if ext.endswith(".zip"):
            try:
                with zipfile.ZipFile(BytesIO(data)) as zf:
                    for f in zf.namelist():
                        if f.endswith((".pt", ".pth")):
                            with zf.open(f) as model_file:
                                model_data = model_file.read()
                                return torch.load(BytesIO(model_data), map_location="cpu"), None
                return None, "ZIP包内未找到 .pt 或 .pth 模型文件"
            except Exception as e2:
                return None, f"解压 ZIP 失败: {e2}"
        # 尝试解压 tar
        elif ext.endswith(".tar") or ext.endswith(".tar.gz") or ext.endswith(".tgz") or ext.endswith(".pth.tar"):
            try:
                with tarfile.open(fileobj=BytesIO(data)) as tf:
                    for member in tf.getmembers():
                        if member.name.endswith((".pt", ".pth")):
                            f = tf.extractfile(member)
                            if f:
                                model_data = f.read()
                                return torch.load(BytesIO(model_data), map_location="cpu"), None
                return None, "tar 包内未找到 .pt 或 .pth 模型文件"
            except Exception as e3:
                return None, f"解压 TAR 失败: {e3}"
        else:
            return None, f"不支持的模型文件格式或加载失败：{e}"

def export_to_onnx_from_file(model_file, input_shape_str, simplify=True):
    if model_file is None:
        return "❌ 请上传模型文件", None

    try:
        input_shape = tuple(int(x) for x in input_shape_str.strip().split(","))
    except Exception as e:
        return f"❌ 输入形状格式错误，应为逗号分隔的整数，如 1,3,224,224。错误：{e}", None

    model, err = load_model_from_bytes(model_file, model_file.name)
    if err:
        return f"❌ 模型加载失败: {err}", None

    # 如果是state_dict，尝试自动载入resnet18
    if isinstance(model, dict):
        try:
            from torchvision.models import resnet18
            base_model = resnet18(pretrained=False)
            base_model.load_state_dict(model)
            model = base_model
        except Exception as e:
            return f"❌ state_dict加载到模型失败: {e}", None

    model.eval()
    dummy_input = torch.randn(input_shape)
    export_path = "exported_model.onnx"

    try:
        torch.onnx.export(model, dummy_input, export_path,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                          opset_version=11)
    except Exception as e:
        return f"❌ ONNX 导出失败：{e}\n{traceback.format_exc()}", None

    try:
        model_onnx = onnx.load(export_path)
        onnx.checker.check_model(model_onnx)

        inputs_info = []
        for inp in model_onnx.graph.input:
            shape = [dim.dim_param or dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            inputs_info.append(f"{inp.name}: {shape}")

        outputs_info = []
        for out in model_onnx.graph.output:
            shape = [dim.dim_param or dim.dim_value for dim in out.type.tensor_type.shape.dim]
            outputs_info.append(f"{out.name}: {shape}")

        info_text = (
            f"✅ 模型成功导出为 {export_path}\n\n"
            "📌 模型输入:\n" + "\n".join(inputs_info) + "\n\n"
            "📌 模型输出:\n" + "\n".join(outputs_info)
        )

    except Exception as e:
        return f"❌ 验证 ONNX 模型失败：{e}", None

    return info_text, export_path


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# PyTorch 模型导出为 ONNX  📦")
        gr.Markdown(
            "上传你的 PyTorch 模型文件，支持 `.pt`、`.pth`、`.zip`、`.tar`、`.pkl`、`.safetensors` 等格式。\n\n"
            "并输入模型输入形状（逗号分隔的整数），"
            "例如 `1,3,224,224` 表示 batch=1，3 通道，224x224 大小。"
        )

        model_file = gr.File(label="上传模型文件", file_types=[".pt", ".pth", ".zip", ".tar", ".pkl", ".safetensors", ".tar.gz", ".tgz", ".pth.tar"])
        input_shape = gr.Textbox(label="输入形状 (逗号分隔整数)", value="1,3,224,224")
        output_area = gr.Textbox(label="导出状态与模型信息", interactive=False, lines=10)
        export_button = gr.Button("导出为 ONNX")

        export_button.click(fn=export_to_onnx_from_file,
                            inputs=[model_file, input_shape],
                            outputs=[output_area, gr.File(label="下载 ONNX 模型")])

    demo.launch()


if __name__ == "__main__":
    main()
