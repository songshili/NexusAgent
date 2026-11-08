"""LlamaIndex + DeepSeek：在 stage3（混合检索 RAG）基础上加入工具调用 + 短期记忆的 Agent。

整体流程（对应下面编号 1~8）：
  1) 配置生成模型（DeepSeek）——负责“决策调用哪个工具”和“读片段写答案”
  2) 配置 Embedding 模型（硅基流动）——负责把文字变成向量
  3) 建索引或加载索引——文档 -> 切分 -> 向量化 -> 落盘
  4) 混合检索——BM25 关键词检索 + 向量语义检索，两路结果融合
  5) 组装带引用的查询引擎（作为 Agent 的一个工具）
  6) 工具封装——RAG 查询引擎包成 QueryEngineTool；日期时间、天气包成 FunctionTool
  7) 短期记忆 + 组装 Agent——ChatMemoryBuffer 让 Agent 记住多轮对话
  8) 交互式对话——逐轮提问，Agent 自主选择工具，并跨轮记住前文

为什么从“查询引擎”升级成“Agent”：
  - stage3 的脚本只能回答知识库里的问题，且每次提问相互独立；
  - Agent 能根据用户意图“自主决定”调用哪个工具：
      知识库问题 -> query_knowledge_base
      今天几号/几点 -> get_current_datetime
      某城市天气   -> get_today_weather
    并通过 Memory 在多轮对话中记住上下文。

密钥通过 .env 读取，不硬编码在代码里。
"""
import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

# --- LlamaIndex 核心组件 ---
from llama_index.core import (
    Settings,                    # 全局配置单例（LLM、embed_model、切分器等）
    SimpleDirectoryReader,       # 从目录读取文档（txt/md/pdf/docx 等）
    StorageContext,              # 存储上下文：管理 docstore / index_store / vector_store
    VectorStoreIndex,            # 向量索引：负责“向量化 + 写入向量库 + 查询”
    load_index_from_storage,     # 从持久化目录恢复索引
)
from llama_index.core.agent import ReActAgent                    # ReAct（文本推理）Agent
from llama_index.core.memory import ChatMemoryBuffer             # 短期对话记忆
from llama_index.core.node_parser import SentenceSplitter        # 按句子/长度切分文档
from llama_index.core.query_engine import CitationQueryEngine    # 生成带 [1][2] 引用的回答
from llama_index.core.retrievers import QueryFusionRetriever     # 融合多个检索器的结果
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES  # 融合算法枚举
from llama_index.core.tools import FunctionTool, QueryEngineTool  # Agent 可调用的工具
from llama_index.embeddings.openai import OpenAIEmbedding        # OpenAI 兼容协议的 embedding 客户端
from llama_index.llms.openai_like import OpenAILike              # OpenAI 兼容协议的 LLM 客户端
from llama_index.retrievers.bm25 import BM25Retriever            # BM25 关键词检索器

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
# 6) 工具封装：把“能力”变成 Agent 可以调用的工具
# ============================================================
# Agent 本身不会检索、不会读时钟；它只会“决定调用哪个工具”。
# 所以我们要把每一种能力都包装成工具（Tool），交给 Agent。
#
# 两类工具的区别：
#   - QueryEngineTool：包装一个查询引擎（这里是整套 RAG 混合检索 + 引用问答）；
#   - FunctionTool：包装一个普通 Python 函数（日期时间、天气）。

# 6.1) RAG 查询引擎 -> QueryEngineTool
#      description 很重要：Agent 靠它判断“什么时候该用这个工具”，
#      所以要写清楚工具能回答哪类问题。
rag_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="query_knowledge_base",
    description=(
        "检索内部知识库（星河知识库平台、青禾社区图书馆智能借阅平台等）回答事实性问题，"
        "返回带 [1][2] 引用编号的答案。涉及产品功能、规则条款、指标、违约后果等问题时使用。"
    ),
)


# 6.2) 获取当前日期和时间 -> FunctionTool（真实函数）
#      FunctionTool 会读取函数签名和类型注解来自动生成参数说明（schema），
#      所以 fn 一定要有明确的类型注解和 docstring。
def get_current_datetime() -> str:
    """返回当前日期和时间，例如 '2026-10-25 20:30:00 Sunday'。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


datetime_tool = FunctionTool.from_defaults(
    fn=get_current_datetime,
    name="get_current_datetime",
    description="获取当前日期和时间。用户问“今天几号”“现在几点”“今天星期几”时使用。",
)


# 6.3) 获取今日天气 -> FunctionTool（模拟数据）
#      这里返回写死的示例天气，保证离线、可重复。真实场景可替换成天气 API 调用。
def get_today_weather(city: str) -> str:
    """返回指定城市今日天气（模拟数据）。"""
    return f"{city} 今日天气：晴，气温 18~26℃，东南风 2 级。（模拟数据，可替换为真实天气 API）"


weather_tool = FunctionTool.from_defaults(
    fn=get_today_weather,
    name="get_today_weather",
    description="获取指定城市今日天气。用户问“XX 今天天气怎么样”时使用，需要传入城市名。",
)

# 所有工具收集成一个列表，稍后交给 Agent。
tools = [rag_tool, datetime_tool, weather_tool]


# ============================================================
# 7) 短期记忆 + 组装 Agent
# ============================================================
# 7.1) 短期记忆：ChatMemoryBuffer 保存最近若干 token 的对话历史。
#      token_limit=4096 表示只保留最近约 4096 个 token 的历史，防止无限增长。
#      注意：在 llama-index 0.14 的 Agent 里，memory 不是构造参数，
#      而是在每次 agent.run() 时作为参数传入——复用同一个对象，历史才会累积。
memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

# 7.2) ReAct Agent：
#      为什么用 ReActAgent 而不是 FunctionAgent？
#      FunctionAgent 依赖模型原生的 function calling，要求 llm 是 FunctionCallingLLM；
#      本项目用的 OpenAILike 不满足该要求，所以改用 ReActAgent（文本式推理）。
#      ReActAgent 在提示词里告诉模型有哪些工具，由模型输出“思考 + 工具调用”，
#      不依赖原生 function calling，兼容任意 LLM，同样能自主选工具。
agent = ReActAgent(
    tools=tools,
    llm=Settings.llm,
    system_prompt=(
        "你是一个中文助手。回答时请遵守："
        "1) 知识库类事实问题（产品功能、规则条款、指标等）必须调用 query_knowledge_base 检索后再回答，并保留 [1][2] 引用；"
        "2) 询问日期或时间时调用 get_current_datetime；"
        "3) 询问某城市天气时调用 get_today_weather；"
        "4) 其他问题直接回答。请结合对话历史理解用户追问。"
    ),
    verbose=True,  # 打印 Agent 每一步的文字与工具调用，方便观察决策过程
)


# ============================================================
# 8) 交互式对话：逐轮提问，观察 Agent 如何选工具、如何记住前文
# ============================================================
# 说明：新版本 Agent 的 run() 返回 WorkflowHandler，内部用 asyncio.create_task
#       启动任务，因此必须在“运行中的事件循环”里调用（见下方 asyncio.run）。
#       await handler 得到 AgentOutput，答案文本在 result.response.content。
async def main() -> None:
    print("\n进入对话模式（输入 exit / quit / 退出 结束）。")
    print("建议按顺序试：")
    print("  1. 星河知识库平台是否支持扫描件 OCR？")
    print("  2. 北京今天天气怎么样？")
    print("  3. 我刚才问的是哪个城市的天气？   <- 验证短期记忆")
    print()

    while True:
        user_input = input("\n你：").strip()
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break
        if not user_input:
            continue

        # 关键：把同一个 memory 对象传进去，Agent 才能看到之前几轮的对话。
        handler = agent.run(user_msg=user_input, memory=memory)
        result = await handler  # result 是 AgentOutput

        print(f"\n助手：{result.response.content}")


if __name__ == "__main__":
    asyncio.run(main())
