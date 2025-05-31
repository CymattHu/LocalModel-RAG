import gradio as gr

from rag_module_ollama import build_vectorstore, answer_query

model_names = ["llama2", "mistral", "qwen"]  # Ollama 模型名示例
compression_methods = ["None"]  # Ollama 内部管理压缩，这里不展示


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

    build_btn.click(build_knowledge_base, inputs=[upload_files], outputs=[gr.Textbox()])
    query_button.click(run_pipeline, inputs=[model_selector, query_input, chatbot], outputs=[chatbot])

demo.launch()
