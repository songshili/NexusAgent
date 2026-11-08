"""LlamaIndex + DeepSeek：Embedding 向量检索 + 混合检索 + 带引用回答 + 索引持久化。

整体流程（对应下面编号 1~6）：
  1) 配置生成模型（DeepSeek）——负责最后“读片段写答案”
  2) 配置 Embedding 模型（硅基流动）——负责把文字变成向量
  3) 建索引或加载索引——文档 -> 切分 -> 向量化 -> 落盘
  4) 混合检索——BM25 关键词检索 + 向量语义检索，两路结果融合
  5) 组装带引用的查询引擎
  6) 逐题提问，打印回答和引用来源

几个关键概念：
  - Embedding（嵌入）：把一段文本映射成一个固定长度的浮点数组（向量）。
    语义相近的文本，向量在空间里的距离也近，所以“向量检索”能按意思而不是按字面匹配。
  - 片段（chunk / node）：整篇文档太长，必须先切成小块再向量化。
    检索时命中的是片段，而不是整篇文档。
  - 索引持久化：把“向量 + 原文片段 + 索引结构”保存成文件，下次启动直接读取，
    避免每次都重新切分和调用 embedding 接口（既慢又费钱）。

密钥通过 .env 读取，不硬编码在代码里。
"""
import os
import re

from dotenv import load_dotenv

# --- LlamaIndex 核心组件 ---
from llama_index.core import (
    Settings,                    # 全局配置单例（LLM、embed_model、切分器等）
    SimpleDirectoryReader,       # 从目录读取文档（txt/md/pdf/docx 等）
    StorageContext,              # 存储上下文：管理 docstore / index_store / vector_store
    VectorStoreIndex,            # 向量索引：负责“向量化 + 写入向量库 + 查询”
    load_index_from_storage,     # 从持久化目录恢复索引
)
from llama_index.core.node_parser import SentenceSplitter   # 按句子/长度切分文档
from llama_index.core.query_engine import CitationQueryEngine  # 生成带 [1][2] 引用的回答
from llama_index.core.retrievers import QueryFusionRetriever   # 融合多个检索器的结果
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES  # 融合算法枚举
from llama_index.embeddings.openai import OpenAIEmbedding      # OpenAI 兼容协议的 embedding 客户端
from llama_index.llms.openai_like import OpenAILike           # OpenAI 兼容协议的 LLM 客户端
from llama_index.retrievers.bm25 import BM25Retriever         # BM25 关键词检索器

# 读取项目根目录下的 .env（KEY=VALUE 形式），把密钥放进环境变量。
load_dotenv()

# 索引持久化目录：运行后会生成 docstore.json / index_store.json / vector_store.json
PERSIST_DIR = "storage"
# 每个检索器各召回多少条、融合后保留多少条。小数据集取 4 足够。
TOP_K = 4


# ============================================================
# 1) 生成模型：DeepSeek
# ============================================================
# Settings 是 LlamaIndex 的全局配置。之后凡是没显式传 llm 的组件，都会用 Settings.llm。
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY，请在 .env 中配置（参考 .env.example）")

Settings.llm = OpenAILike(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    api_base="https://api.deepseek.com",   # DeepSeek 的 OpenAI 兼容地址
    api_key=api_key,
    is_chat_model=True,  # 强制走 /chat/completions（DeepSeek 不支持 legacy /completions）
)


# ============================================================
# 2) Embedding 模型：硅基流动 BAAI/bge-m3
# ============================================================
# DeepSeek 只提供“对话”接口，不提供 embedding 接口，所以向量化另用硅基流动。
# 硅基流动的接口协议与 OpenAI 完全兼容，因此直接用 OpenAIEmbedding 客户端，
# 只把 api_base 换成硅基流动的地址，不需要额外写 HTTP 请求。
embed_api_key = os.getenv("EMBED_API_KEY")
if not embed_api_key:
    raise SystemExit("未找到 EMBED_API_KEY，请在 .env 中配置（参考 .env.example）")

# 注意：llama-index-embeddings-openai 0.7.0 必须用 model_name=
# （用 model= 传自定义模型名会抛 ValueError，它只认 OpenAI 官方模型枚举）。
Settings.embed_model = OpenAIEmbedding(
    model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    api_base=os.getenv("EMBED_API_BASE", "https://api.siliconflow.cn/v1"),
    api_key=embed_api_key,
)


# ============================================================
# 3) 索引：首次构建并持久化，之后直接加载
# ============================================================
# storage/ 是否存在，就代表“索引是否已经建过”。
# 注意：这里没有做“文档变更检测”。如果修改了 data/ 里的文件，
# 需要手动删除 storage/ 重新构建，否则加载的仍是旧索引。
if os.path.exists(PERSIST_DIR):
    print(f"检测到已有索引，直接从 {PERSIST_DIR}/ 加载……")
    # StorageContext.from_defaults 会自动读取目录下的各个 json 文件，
    # 恢复 docstore（原文片段）和 vector_store（向量）。
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("未找到已有索引，开始构建（加载文档 -> 切分 -> embedding）……")

    # 3.1) 读取 data/ 下所有支持的文档，每篇变成一个 Document 对象。
    documents = SimpleDirectoryReader("data").load_data()

    # 3.2) 切分：chunk_size=512 表示每段约 512 个 token，chunk_overlap=64 表示
    #      相邻片段重叠 64 个 token。重叠是为了避免答案刚好被切断在边界上。
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(
        documents
    )

    # 3.3) 建向量索引：对每个 node 调用 embedding 接口，把向量写进向量库，
    #      同时把原文片段写进 docstore（后面 BM25 检索要用到原文）。
    index = VectorStoreIndex(nodes)

    # 3.4) 落盘。persist 之后，下次启动就能走上面的 load 分支。
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"索引已构建并持久化到 {PERSIST_DIR}/（共 {len(nodes)} 个片段）")


# ============================================================
# 4) 混合检索：BM25 关键词 + 向量语义
# ============================================================
# 为什么要混合：
#   - BM25 擅长精确词匹配（专有名词、编号、数字，如“版本 3.1”“P95”）；
#   - 向量检索擅长语义相近但用词不同的情况（如“违约”与“未按时取书”）。
#   两者互补，融合后召回更稳。

# 4.1) BM25 检索器：传统关键词检索，基于词频统计，不需要模型。
#      index=index 表示直接从索引的 docstore 里取原文片段，不用另外传 nodes。
bm25_retriever = BM25Retriever.from_defaults(
    index=index,
    similarity_top_k=TOP_K,
    # 分词规则：英文/数字按连续串切，中文按“单字”切。
    # 中文没有空格，先按单字切是最简单可靠的做法（配合 language="zh" 去停用词）。
    token_pattern=r"[a-zA-Z0-9]+|[一-鿿]",
    language="zh",
)

# 4.2) 向量检索器：把问题也 embedding 成向量，在向量库里找最相近的 TOP_K 个片段。
vector_retriever = index.as_retriever(similarity_top_k=TOP_K)

# 4.3) 融合检索器：同时调用上面两个检索器，再把两份排名合成一份。
retriever = QueryFusionRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    similarity_top_k=TOP_K,                  # 融合后最终保留 TOP_K 条
    mode=FUSION_MODES.RECIPROCAL_RANK,       # Reciprocal Rank Fusion（倒数排名融合，见下）
    num_queries=1,  # 不做 LLM 查询改写，只融合两路检索结果
    use_async=False,  # 同步执行，脚本里更直观（异步对结果没有影响）
)
# RRF 的直觉：某条结果在任一路检索里排得越靠前，加分越多；
# 最终得分 ≈ Σ 1/(常数 + 该路排名)。两路都靠前的结果会脱颖而出。
# 好处是不用关心 BM25 分数和向量余弦分数“量纲不同”的问题，只看排名。


# ============================================================
# 5) 带引用的查询引擎
# ============================================================
# CitationQueryEngine 在把检索结果交给 LLM 之前，会给每个片段加上编号
# （内部形如 “Source 1: ...”、“Source 2: ...”），并在提示词里要求模型
# 用 [1]、[2] 标注出处。所以最终回答里的 [N] 与 response.source_nodes 的顺序一一对应。
#
# citation_chunk_size / citation_chunk_overlap：引用粒度的二次切分参数。
# 这里与建索引时的 512/64 保持一致，避免一个片段被拆得过于零碎。
query_engine = CitationQueryEngine.from_args(
    index=index,
    retriever=retriever,       # 用上面组装好的混合检索器
    llm=Settings.llm,
    citation_chunk_size=512,
    citation_chunk_overlap=64,
)


# ============================================================
# 6) 逐题提问，打印回答与引用来源
# ============================================================
# 注：一个问题只问一件事，检索召回更准；
# 把多个不相关子问题拼在一起会稀释语义，导致相关片段排不进 top-k。
questions = [
    "星河知识库平台是否支持扫描件 OCR 和图片内容识别？",
    "青禾社区图书馆智能借阅平台 30 天内累计 2 次违约会有什么后果？暂停期满后是否需要人工申请恢复？",
    "检索接口的 P95 响应时间目标和单实例并发上限分别是多少？",
]

for question in questions:
    # query() 内部依次完成：检索 -> 给片段编号 -> 拼提示词 -> 调 DeepSeek -> 解析引用
    response = query_engine.query(question)

    print(f"\n{'=' * 60}")
    print(f"问题：{question}\n")
    print(f"回答：{response}\n")
    print("引用来源：")

    # response.source_nodes 就是回答里 [N] 引用的片段列表：
    # 第 1 个元素对应 [1]，第 2 个对应 [2]，以此类推。
    for i, source_node in enumerate(response.source_nodes, 1):
        node = source_node.node
        # 片段所属文件名，来自 SimpleDirectoryReader 自动写入的元数据。
        file_name = node.metadata.get("file_name", "未知文件")
        # 这里显示的是融合后的 RRF 分数（不是向量相似度），只用于观察排序。
        score = f"{source_node.score:.4f}" if source_node.score is not None else "-"
        # CitationQueryEngine 会在每个引用块前加 "Source N:"，打印时去掉
        snippet = re.sub(r"^Source \d+:\s*", "", node.get_content())
        snippet = snippet.replace("\n", " ")[:200]
        print(f"  [{i}] {file_name} (score={score})")
        print(f"      {snippet}……")
