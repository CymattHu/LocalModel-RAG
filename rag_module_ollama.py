from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from ollama import chat
from langchain_huggingface import HuggingFaceEmbeddings



def build_vectorstore(file_objs):
    docs = []
    for file in file_objs:
        if file.name.endswith(".pdf"):
            docs += PyPDFLoader(file.name).load()
        else:
            docs += TextLoader(file.name).load()

    # embedding = HuggingFaceEmbeddings()
    embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    vs = FAISS.from_documents(docs, embedding)
    return vs

def answer_query(model_name, vectorstore, question):
    # 使用 retriever 的新写法（invoke）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    # 拼接文档内容
    context = "\n\n".join([doc.page_content for doc in docs])

    # 构造提示词
    prompt = f"根据以下内容回答问题：\n{context}\n\n问题：{question}\n回答："

    # 本地调用 Ollama 模型
    response = chat(model=model_name, messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response.message.content.strip()
