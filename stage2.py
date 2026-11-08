"""LlamaIndex + DeepSeek 最小 RAG 示例。

流程：加载文档 -> 切分 -> BM25 关键词检索 -> DeepSeek 生成回答。
密钥通过 .env 文件读取，不硬编码在代码里。
"""
import os

from dotenv import load_dotenv

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY，请在 .env 中配置（参考 .env.example）")

# 1) 配置生成模型（DeepSeek）
Settings.llm = OpenAILike(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    api_base="https://api.deepseek.com",
    api_key=api_key,
    is_chat_model=True,  # 强制走 /chat/completions（DeepSeek 不支持 legacy /completions）
)

# 2) 加载文档并切分
documents = SimpleDirectoryReader("data").load_data()
nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(documents)

# 3) BM25 关键词检索（中文按单字、英文按单词分词，配合中文停用词）
retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=4,
    token_pattern=r"[a-zA-Z0-9]+|[一-鿿]",
    language="zh",
)

# 4) 组装查询引擎并提问
query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=Settings.llm)

question = "星河知识库平台是否支持扫描件 OCR 和图片内容识别？青禾社区图书馆智能借阅平台30 天内累计 2 次违约会有什么后果？暂停期满后是否需要人工申请恢复？"
response = query_engine.query(question)
print(f"问题：{question}\n")
print(f"回答：{response}\n")
