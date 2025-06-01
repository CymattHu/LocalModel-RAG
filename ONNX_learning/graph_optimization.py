import onnx
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from onnxoptimizer import optimize

def optimize_onnx_model(input_path: str, output_path: str):
    # 1. 加载模型
    model = onnx.load(input_path)
    print("模型加载成功。")

    # 2. 使用 onnxoptimizer 进行多种优化pass
    passes = [
        "eliminate_deadend",                  # 删除无用节点
        "eliminate_identity",                 # 删除恒等节点
        "eliminate_nop_dropout",              # 删除无操作dropout
        "eliminate_nop_transpose",            # 删除无操作转置
        "fuse_add_bias_into_conv",            # 卷积加偏置融合
        "fuse_bn_into_conv",                  # BN融合进卷积
        "fuse_consecutive_transposes",        # 连续转置融合
        "fuse_matmul_add_bias_into_gemm",    # matmul+加偏置融合成gemm
        "fuse_pad_into_conv",                 # pad融合
        "fuse_transpose_into_gemm",           # transpose融合
        "eliminate_unused_initializer"        # 删除没用到的initializer
    ]

    print("正在应用onnxoptimizer优化Pass...")
    optimized_model = optimize(model, passes)
    print("onnxoptimizer优化完成。")

    # 3. 保存优化后模型
    onnx.save(optimized_model, output_path)
    print(f"优化后的模型已保存至: {output_path}")

    # 4. 用 ONNX Runtime 验证模型有效性和开启Graph Optimization
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    try:
        session = InferenceSession(output_path, sess_options)
        print("ONNX Runtime成功加载并验证模型，且开启扩展优化。")
    except Exception as e:
        print(f"ONNX Runtime加载模型失败: {e}")

if __name__ == "__main__":
    input_onnx = "resnet18.onnx"
    output_onnx = "model_optimized.onnx"
    optimize_onnx_model(input_onnx, output_onnx)
