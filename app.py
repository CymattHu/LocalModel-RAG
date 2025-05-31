import gradio as gr
from rag_module_ollama import build_vectorstore, answer_query

model_names = ["llama2", "mistral", "qwen"]

vectorstore = None

def build_knowledge_base(files):
    global vectorstore
    vectorstore = build_vectorstore(files)
    return "知识库构建完成！"

def run_pipeline(model_name, query, history=[]):
    global vectorstore
    if vectorstore is None:
        return history + [("系统", "请先上传并构建知识库")]

    response = answer_query(model_name, vectorstore, query)
    history.append((query, response))
    return history

# 新增：一个测试接口，方便 CI 调用
def test_app():
    # 这里写你能自动提供的测试数据
    dummy_files = ["dummy_files/test.pdf"]  # 这里换成你的测试文档路径或者模拟内容
    dummy_query = "测试问题？"
    dummy_model = "llama2"

    # 构建知识库
    try:
        build_knowledge_base(dummy_files)
    except Exception as e:
        return f"构建知识库失败: {e}"

    # 查询
    try:
        history = run_pipeline(dummy_model, dummy_query, [])
        if history and "请先上传" not in history[-1][1]:
            return "测试通过"
        else:
            return "测试失败：未正确返回结果"
    except Exception as e:
        return f"查询失败: {e}"

# 下面是原来UI启动的代码
with gr.Blocks() as demo:
    gr.Markdown("### Ollama + LangChain 本地轻量化推理平台 Demo")

    with gr.Row():
        model_selector = gr.Dropdown(model_names, label="选择模型")

    with gr.Row():
        upload_files = gr.File(file_types=[".pdf", ".txt", ".md"], label="上传知识库文档", file_count="multiple")
        build_btn = gr.Button("构建知识库")

    with gr.Row():
        query_input = gr.Textbox(label="输入问题")
        query_button = gr.Button("查询")

    chatbot = gr.Chatbot()

    build_btn.click(build_knowledge_base, inputs=[upload_files], outputs=[gr.Textbox()])
    query_button.click(run_pipeline, inputs=[model_selector, query_input, chatbot], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()