"""LlamaIndex + DeepSeek：在 stage4（工具调用 Agent）基础上升级为“研究模式”。

相对 stage4 的四个变化（对应下面编号 1~9）：
  1) 模型统一改为 deepseek-flash
  2) 工具调用鲁棒性：
       - 工具内部抛异常不会让 Agent 崩溃（框架会把异常转成错误观察值，Agent 继续推理）
       - 空结果返回明确的“下一步”建议，而不是让模型瞎猜
       - 进程内第一次检索会被人为注入一次故障，用来演示“Agent 不崩溃且能向用户说明”
  3) 短期记忆从 ChatMemoryBuffer 迁移到 Memory（llama-index 0.14 的新记忆组件）
  4) 新增 list_files 工具：Agent 可以先看 data/ 里有哪些资料，再决定检索哪个文件

研究模式流水线（research_topic 工具）：
    主题 -> LLM 拆成多个检索角度 -> 每个角度做 BM25+向量混合检索
         -> 按 node_id 去重、按命中次数/分数筛选
         -> LLM 合成结构化报告（每条事实带 [n] 引用）
         -> 程序追加“引用来源”清单（[n] -> 文件名 + 片段）

整体流程（对应下面编号 1~9）：
  1) 配置生成模型（deepseek-flash）
  2) 配置 Embedding 模型（硅基流动 BAAI/bge-m3）
  3) 建索引或加载索引
  4) 混合检索器（全局 + 按文件）
  5) 检索鲁棒性：模拟故障 + 空结果引导
  6) 研究模式流水线
  7) 工具封装：list_files / query_knowledge_base / research_topic / 日期 / 天气
  8) Memory + ReAct Agent
  9) 交互式对话

密钥通过 .env 读取，不硬编码在代码里。
"""
import asyncio
import os
import re
import threading
from collections import Counter
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
from llama_index.core.llms import ChatMessage, MessageRole      # 直接调用 LLM 时用
from llama_index.core.memory import Memory                      # 新版短期记忆（替代 ChatMemoryBuffer）
from llama_index.core.node_parser import SentenceSplitter        # 按句子/长度切分文档
from llama_index.core.retrievers import QueryFusionRetriever     # 融合多个检索器的结果
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES  # 融合算法枚举
from llama_index.core.schema import NodeWithScore                # 带分数的检索片段
from llama_index.core.tools import FunctionTool                  # 把普通 Python 函数包成工具
from llama_index.core.vector_stores import (                     # 向量检索的元数据过滤
    ExactMatchFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding        # OpenAI 兼容协议的 embedding 客户端
from llama_index.llms.openai_like import OpenAILike              # OpenAI 兼容协议的 LLM 客户端
from llama_index.retrievers.bm25 import BM25Retriever            # BM25 关键词检索器

# 读取项目根目录下的 .env（KEY=VALUE 形式），把密钥放进环境变量。
load_dotenv()

PERSIST_DIR = "storage"   # 索引持久化目录
DATA_DIR = "data"         # 资料目录
TOP_K = 4                 # 每个检索器 / 每个角度召回多少条
RESEARCH_TOP_K = 6        # 研究报告最终使用多少条资料
MAX_SUBQUERIES = 4        # 最多生成几个检索角度
TOKEN_PATTERN = r"[a-zA-Z0-9]+|[一-鿿]"  # 英文/数字按词，中文按单字
# 研究模式的报告很长，token_limit 太小会把历史整段刷掉；
# 30000 足够保留若干轮“研究主题 + 报告”，避免追问时失忆。
MEMORY_TOKEN_LIMIT = 30000

# 是否在进程内第一次检索时人为注入一次故障（1/0，默认开启）。
SIMULATE_RETRIEVAL_FAILURE = os.getenv(
    "SIMULATE_RETRIEVAL_FAILURE", "1"
).strip().lower() not in {"0", "false", "no", "off", ""}


# ============================================================
# 1) 生成模型：DeepSeek deepseek-flash
# ============================================================
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY，请在 .env 中配置（参考 .env.example）")

Settings.llm = OpenAILike(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
    api_base="https://api.deepseek.com",   # DeepSeek 的 OpenAI 兼容地址
    api_key=api_key,
    is_chat_model=True,  # 强制走 /chat/completions
)


# ============================================================
# 2) Embedding 模型：硅基流动 BAAI/bge-m3
# ============================================================
# DeepSeek 只提供“对话”接口，不提供 embedding 接口，所以向量化另用硅基流动。
# 注意：llama-index-embeddings-openai 0.7.0 必须用 model_name=
# （用 model= 传自定义模型名会抛 ValueError，它只认 OpenAI 官方模型枚举）。
embed_api_key = os.getenv("EMBED_API_KEY")
if not embed_api_key:
    raise SystemExit("未找到 EMBED_API_KEY，请在 .env 中配置（参考 .env.example）")

Settings.embed_model = OpenAIEmbedding(
    model_name=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
    api_base=os.getenv("EMBED_API_BASE", "https://api.siliconflow.cn/v1"),
    api_key=embed_api_key,
)


# ============================================================
# 3) 索引：首次构建并持久化，之后直接加载
# ============================================================
# 与 stage3/stage4 共用 storage/。embedding 模型没变，已有索引可以直接复用。
if os.path.exists(PERSIST_DIR):
    print(f"检测到已有索引，直接从 {PERSIST_DIR}/ 加载……")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    print("未找到已有索引，开始构建（加载文档 -> 切分 -> embedding）……")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64).get_nodes_from_documents(
        documents
    )
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"索引已构建并持久化到 {PERSIST_DIR}/（共 {len(nodes)} 个片段）")


# ============================================================
# 4) 混合检索器：全局 + 按文件
# ============================================================
# 4.1) 全局 BM25：从整个索引的 docstore 里取原文片段做关键词检索。
base_bm25 = BM25Retriever.from_defaults(
    index=index,
    similarity_top_k=TOP_K,
    token_pattern=TOKEN_PATTERN,
    language="zh",
)

# 4.2) 全局向量检索：把问题 embedding 后在向量库里找最相近的片段。
base_vector = index.as_retriever(similarity_top_k=TOP_K)

# 4.3) 全局融合检索：两路结果按 RRF（倒数排名）合成一份。
base_retriever = QueryFusionRetriever(
    retrievers=[base_bm25, base_vector],
    similarity_top_k=TOP_K,
    mode=FUSION_MODES.RECIPROCAL_RANK,
    num_queries=1,     # 不做 LLM 查询改写
    use_async=False,   # 工具是同步函数，避免在 Agent 的事件循环里嵌套 asyncio
)

# 4.4) 按文件检索器缓存：Agent 选定文件后，只在该文件内检索。
#      做法：BM25 用该文件的 nodes 重建；向量检索用 file_name 元数据过滤。
_file_retriever_cache: dict[str, QueryFusionRetriever] = {}


def _indexed_file_names() -> set[str]:
    """返回已经写入索引的文件名集合。"""
    return {
        str(node.metadata["file_name"])
        for node in index.docstore.docs.values()
        if node.metadata.get("file_name")
    }


def _available_file_names() -> list[str]:
    """返回 data/ 目录和索引里出现过的所有文件名（用于错误提示）。"""
    names = set(_indexed_file_names())
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, name)
            if not name.startswith(".") and os.path.isfile(path):
                names.add(name)
    return sorted(names)


def _resolve_file_name(file_name: str) -> tuple[str | None, str]:
    """把用户/模型给的文件名解析成索引里的规范文件名。

    返回 (规范文件名, 错误提示)：
      - 未指定文件：("", "")
      - 解析成功：("RAG.txt", "")
      - 解析失败：(None, 给 Agent 看的下一步建议)
    """
    raw = (file_name or "").strip().strip("'\"")
    if not raw:
        return "", ""

    # 允许模型传 data/RAG.txt 或 RAG.txt
    normalized = os.path.basename(raw.replace("\\", "/"))
    available = _available_file_names()

    matched: str | None = None
    if normalized in available:
        matched = normalized
    else:
        # 忽略大小写
        lower_map = {name.lower(): name for name in available}
        if normalized.lower() in lower_map:
            matched = lower_map[normalized.lower()]
        else:
            # 唯一的主文件名匹配，例如 RAG -> RAG.txt
            stem = os.path.splitext(normalized)[0].lower()
            candidates = [
                name
                for name in available
                if os.path.splitext(name)[0].lower() == stem
            ]
            if len(candidates) == 1:
                matched = candidates[0]

    if matched is None:
        names = "、".join(available) if available else "（无）"
        return None, (
            f"未找到文件“{normalized}”。当前可用文件：{names}。\n"
            "下一步建议：\n"
            "1) 先调用 list_files 查看 data/ 中的资料清单；\n"
            "2) 使用清单里的完整文件名重试；\n"
            "3) 或者不传 file_name，检索全部资料。"
        )

    if matched not in _indexed_file_names():
        return None, (
            f"文件“{matched}”在 {DATA_DIR}/ 中存在，但尚未写入索引。\n"
            f"下一步建议：删除 {PERSIST_DIR}/ 后重新运行 stage5.py 重建索引，再重试。"
        )

    return matched, ""


def _get_retriever(file_name: str) -> QueryFusionRetriever | None:
    """按文件名返回检索器；file_name 为空表示全局检索。"""
    if not file_name:
        return base_retriever
    if file_name in _file_retriever_cache:
        return _file_retriever_cache[file_name]

    nodes = [
        node
        for node in index.docstore.docs.values()
        if node.metadata.get("file_name") == file_name
    ]
    if not nodes:
        return None

    # BM25 的 top_k 不能超过语料数量，否则会告警并自动截断。
    k = max(1, min(TOP_K, len(nodes)))
    bm25 = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=k,
        token_pattern=TOKEN_PATTERN,
        language="zh",
    )
    vector = index.as_retriever(
        similarity_top_k=k,
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="file_name", value=file_name)]
        ),
    )
    fused = QueryFusionRetriever(
        retrievers=[bm25, vector],
        similarity_top_k=k,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        num_queries=1,
        use_async=False,
    )
    _file_retriever_cache[file_name] = fused
    return fused


# ============================================================
# 5) 检索鲁棒性：模拟故障 + 空结果引导
# ============================================================
_retrieval_call_count = 0
_retrieval_lock = threading.Lock()
# 注入故障后置为 True；下一次成功检索时由工具把“已重试成功”的说明带给 Agent。
_failure_notice_pending = False
# 累计注入次数：用于在交互层确保用户一定看到“发生过模拟故障”的说明。
_simulated_failure_events = 0


def _hybrid_retrieve(
    query: str, file_name: str = "", top_k: int = TOP_K
) -> list[NodeWithScore]:
    """统一的混合检索入口。

    人为故障：进程内第一次调用时抛 RuntimeError。这不是 bug，而是用来演示：
      - 工具抛异常时，LlamaIndex 的 AgentWorkflow 会把异常转成
        ToolOutput(is_error=True)，Agent 不会崩溃，而是继续推理；
      - 系统提示要求 Agent 先向用户说明故障，再重试一次；
      - 第二次调用就恢复正常。
    可用环境变量 SIMULATE_RETRIEVAL_FAILURE=0 关闭。
    """
    global _retrieval_call_count, _failure_notice_pending, _simulated_failure_events
    with _retrieval_lock:
        _retrieval_call_count += 1
        should_fail = (
            SIMULATE_RETRIEVAL_FAILURE and _retrieval_call_count == 1
        )
        if should_fail:
            _failure_notice_pending = True
            _simulated_failure_events += 1

    if should_fail:
        raise RuntimeError(
            "SIMULATED_RETRIEVAL_FAILURE（stage5 演示）：检索服务暂时不可用，"
            "本次检索被人为注入了一次故障，尚未执行。这是可恢复的模拟错误："
            "请向用户说明发生了模拟故障，并立即重试同一个工具一次；重试会成功。"
        )

    retriever = _get_retriever(file_name)
    if retriever is None:
        return []
    return retriever.retrieve(query)[:top_k]


def _consume_failure_notice() -> str:
    """若之前注入过故障，返回一段“已重试成功”的说明，供 Agent 转达给用户。"""
    global _failure_notice_pending
    with _retrieval_lock:
        if not _failure_notice_pending:
            return ""
        _failure_notice_pending = False
    return (
        "【系统说明】上一次检索调用被人为注入了一次 SIMULATED_RETRIEVAL_FAILURE；"
        "本次是自动重试，已经成功。请在最终回答开头用一句话向用户说明："
        "“检索过程中出现一次模拟故障，已重试成功”。\n\n"
    )


def _snippet(text: str, limit: int) -> str:
    """压缩空白并截断，方便放进提示词或打印。"""
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _format_sources(nodes: list[NodeWithScore], max_chars: int = 600) -> str:
    """把检索片段格式化成带 [n] 编号的资料块。"""
    blocks = []
    for i, item in enumerate(nodes, 1):
        file_name = item.node.metadata.get("file_name", "未知文件")
        score = f"{item.score:.4f}" if item.score is not None else "-"
        blocks.append(
            f"[{i}] 文件：{file_name}（score={score}，仅用于排序）\n"
            f"{_snippet(item.node.get_content(), max_chars)}"
        )
    return "\n\n".join(blocks)


def _empty_result_guidance(query: str, file_name: str = "") -> str:
    """空结果不是终点：明确告诉 Agent 下一步可以做什么。"""
    scope = f"文件 {file_name} 中" if file_name else "知识库中"
    available = "、".join(_available_file_names()) or "（无）"
    lines = [
        f"未检索到与“{query}”相关的内容（{scope}）。",
        "下一步建议：",
        "1) 先调用 list_files 查看 data/ 里有哪些资料，确认主题是否在知识库覆盖范围内；",
        "2) 换用更具体的关键词、专有名词、产品名或指标名重试；",
    ]
    if file_name:
        lines.append("3) 去掉 file_name 参数，改为检索全部资料。")
    else:
        lines.append("3) 如果多次更换关键词仍无结果，说明知识库可能没有该主题，应向用户如实说明。")
    lines.append(f"当前可用文件：{available}")
    return "\n".join(lines)


# ============================================================
# 6) 研究模式流水线
# ============================================================
def _llm_text(prompt: str) -> str:
    """直接调用一次 LLM，返回纯文本。

    用 chat() 显式走 /chat/completions（DeepSeek 不支持 legacy /completions）。
    """
    response = Settings.llm.chat(
        [ChatMessage(role=MessageRole.USER, content=prompt)]
    )
    return (response.message.content or "").strip()


SUBQUERY_PROMPT = """你是检索规划助手。请把下面的研究主题拆成 3~4 个互补的中文检索子问题，
覆盖：定义/背景、关键规则或指标、问题/风险/处理、对比/差异（如适用）。
要求：
- 每行只输出一个子问题；
- 不要编号、不要解释、不要输出主题之外的内容；
- 子问题要短，便于关键词检索和向量检索。
{scope_hint}
研究主题：{topic}"""

REPORT_PROMPT = """你是一名严谨的中文研究助理。请只根据下面提供的资料，围绕主题写一份结构化研究报告。

主题：{topic}

资料（编号 [1]...[n]，每段标注了文件名）：
{context}

写作要求：
1. 先用 2~4 条“核心结论”概括；
2. 再按角度分节分析（定义/机制、规则/指标、问题/风险、差异/建议），没有资料的角度不要写；
3. 每一个事实性结论后面必须用 [n] 标注来源，n 必须来自上面的资料编号；同一结论可标多个，如 [1][3]；
4. 只能使用资料中的信息，不得编造；资料不足时明确写“现有资料不足以判断”；
5. 不要单独输出“引用来源”列表（程序会自动追加）；
6. 用 Markdown 输出，语言为中文。"""

# 子问题按 (topic, file_name) 缓存：模拟故障后 Agent 重试时不必重新生成。
_subquery_cache: dict[tuple[str, str], list[str]] = {}


def _generate_subqueries(topic: str, file_name: str = "") -> list[str]:
    """把研究主题拆成多个检索角度；LLM 失败时回退到固定角度。"""
    cache_key = (topic, file_name)
    if cache_key in _subquery_cache:
        return _subquery_cache[cache_key]

    fallback = [
        topic,
        f"{topic} 定义 功能 背景",
        f"{topic} 规则 指标 数据",
        f"{topic} 问题 风险 故障 处理",
    ]

    scope_hint = f"限定资料文件：{file_name}" if file_name else "资料范围：全部文件"
    try:
        text = _llm_text(
            SUBQUERY_PROMPT.format(topic=topic, scope_hint=scope_hint)
        )
        queries: list[str] = []
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", line).strip()
            if line and len(line) <= 100:
                queries.append(line)
        queries = list(dict.fromkeys(queries))[:MAX_SUBQUERIES]
        if len(queries) < 2:
            queries = fallback[:MAX_SUBQUERIES]
        elif topic not in queries:
            # 保底：原始主题本身也是一个角度
            queries = [topic, *queries][:MAX_SUBQUERIES]
    except Exception as exc:  # LLM 网络/接口异常不影响研究流程
        print(f"[stage5] 子问题生成失败，改用固定角度：{exc}")
        queries = fallback[:MAX_SUBQUERIES]

    _subquery_cache[cache_key] = queries
    return queries


def _synthesize_report(topic: str, sources: list[NodeWithScore]) -> str:
    """把筛选后的资料交给 LLM，合成带 [n] 引用的报告。失败返回空串。"""
    context = "\n\n".join(
        f"[{i}] 文件：{item.node.metadata.get('file_name', '未知文件')}\n"
        f"{_snippet(item.node.get_content(), 800)}"
        for i, item in enumerate(sources, 1)
    )
    try:
        return _llm_text(REPORT_PROMPT.format(topic=topic, context=context))
    except Exception as exc:
        print(f"[stage5] 报告合成失败，返回原始资料：{exc}")
        return ""


def _append_citations(report: str, sources: list[NodeWithScore]) -> str:
    """程序追加“引用来源”，保证 [n] 与文件名/片段一一对应。"""
    lines = ["", "## 引用来源"]
    for i, item in enumerate(sources, 1):
        file_name = item.node.metadata.get("file_name", "未知文件")
        lines.append(
            f"[{i}] {file_name} — {_snippet(item.node.get_content(), 120)}…"
        )
    return report.rstrip() + "\n" + "\n".join(lines)


def research_topic(topic: str, file_name: str = "") -> str:
    """研究模式：多角度检索并生成带 [n] 引用的结构化研究报告。

    Args:
        topic: 研究主题，例如“星河知识库平台的扫描件处理能力”。
        file_name: 可选，限定只研究某个文件；跨文件研究请留空。
    """
    topic = (topic or "").strip()
    if not topic:
        return "研究主题为空。下一步建议：重新输入一个具体主题后重试。"

    canonical, error = _resolve_file_name(file_name)
    if error:
        return error

    subqueries = _generate_subqueries(topic, canonical)

    # 6.1) 多角度检索：同一片段被多个角度命中，说明它更可能是核心资料。
    aggregated: dict[str, dict] = {}
    failure_notice = ""
    for query in subqueries:
        # 这里故意不捕获异常：模拟故障会抛给 Agent 框架，由框架兜底并让 Agent 重试。
        results = _hybrid_retrieve(query, canonical, TOP_K)
        if not failure_notice:
            failure_notice = _consume_failure_notice()
        for item in results:
            content = item.node.get_content().strip()
            if len(content) < 30:  # 过滤过短/空片段
                continue
            entry = aggregated.setdefault(
                item.node.node_id,
                {"node": item.node, "hits": 0, "best": 0.0},
            )
            entry["hits"] += 1
            score = item.score or 0.0
            if score > entry["best"]:
                entry["best"] = score

    if not aggregated:
        return failure_notice + _empty_result_guidance(topic, canonical)

    # 6.2) 筛选：命中次数优先，其次最佳分数；只保留前 RESEARCH_TOP_K 条。
    ranked = sorted(
        aggregated.values(), key=lambda entry: (-entry["hits"], -entry["best"])
    )[:RESEARCH_TOP_K]
    sources = [
        NodeWithScore(node=entry["node"], score=entry["best"])
        for entry in ranked
    ]

    # 6.3) 总结：一次 LLM 调用生成结构化报告；失败则退回原始资料。
    report = _synthesize_report(topic, sources)
    header = (
        f"> 检索角度：{'；'.join(subqueries)}"
        f"（共 {len(sources)} 条资料）\n\n"
    )
    if not report:
        return (
            failure_notice
            + header
            + "（报告生成失败，以下为检索到的原始资料，供人工整理）\n\n"
            + _format_sources(sources, 800)
        )
    return failure_notice + header + _append_citations(report, sources)


# ============================================================
# 7) 工具封装
# ============================================================
# Agent 本身不会检索、不会读目录；它只会“决定调用哪个工具”。
def list_files() -> str:
    """列出 data/ 目录中可供检索的资料文件（文件名、大小、索引片段数）。"""
    if not os.path.isdir(DATA_DIR):
        return (
            f"目录 {DATA_DIR}/ 不存在。\n"
            f"下一步建议：创建 {DATA_DIR}/ 并放入资料文件后重试。"
        )

    indexed_counts = Counter(
        str(node.metadata.get("file_name"))
        for node in index.docstore.docs.values()
        if node.metadata.get("file_name")
    )

    lines = []
    try:
        for name in sorted(os.listdir(DATA_DIR)):
            path = os.path.join(DATA_DIR, name)
            if name.startswith(".") or not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
            count = indexed_counts.get(name, 0)
            status = (
                f"已索引，{count} 个片段"
                if count
                else f"尚未索引（需删除 {PERSIST_DIR}/ 重建）"
            )
            lines.append(f"- {name}（{size} 字节）— {status}")
    except OSError as exc:
        return (
            f"读取 {DATA_DIR}/ 失败：{exc}\n"
            "下一步建议：检查目录是否存在、是否有读取权限后重试。"
        )

    if not lines:
        return (
            f"{DATA_DIR}/ 目录下没有可用资料。\n"
            f"下一步建议：放入 .txt 等文档并重建索引（删除 {PERSIST_DIR}/ 后重跑）。"
        )

    return (
        f"{DATA_DIR}/ 目录下的资料文件：\n"
        + "\n".join(lines)
        + "\n\n下一步：把完整文件名作为 file_name 传给 query_knowledge_base 或 "
        "research_topic；跨文件研究不要传 file_name。"
    )


def query_knowledge_base(query: str, file_name: str = "") -> str:
    """在知识库中做一次 BM25+向量混合检索，返回带 [n] 编号的原文片段。

    Args:
        query: 检索问题或关键词。
        file_name: 可选，限定只检索某个文件；跨文件检索请留空。
    """
    query = (query or "").strip()
    if not query:
        return "检索问题为空。下一步建议：输入具体问题或关键词后重试。"

    canonical, error = _resolve_file_name(file_name)
    if error:
        return error

    # 模拟故障会从这里抛出；框架会捕获，Agent 不会崩溃。
    nodes = _hybrid_retrieve(query, canonical, TOP_K)
    failure_notice = _consume_failure_notice()
    if not nodes:
        return failure_notice + _empty_result_guidance(query, canonical)

    return (
        failure_notice
        + "检索到以下片段（score 仅用于排序，不代表置信度）。"
        "请基于片段回答，保留 [n] 引用，不要编造片段之外的信息。\n\n"
        + _format_sources(nodes)
    )


# 7.1) 获取当前日期和时间
def get_current_datetime() -> str:
    """返回当前日期和时间，例如 '2026-10-25 20:30:00 Sunday'。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")


# 7.2) 获取今日天气（模拟数据）
def get_today_weather(city: str) -> str:
    """返回指定城市今日天气（模拟数据）。"""
    return f"{city} 今日天气：晴，气温 18~26℃，东南风 2 级。（模拟数据，可替换为真实天气 API）"


tools = [
    FunctionTool.from_defaults(
        fn=list_files,
        name="list_files",
        description=(
            "列出 data/ 中所有可检索资料文件（文件名、大小、索引片段数）。"
            "当用户问“有哪些资料/文件”，或你不确定该检索哪个文件时，先调用本工具。"
        ),
    ),
    FunctionTool.from_defaults(
        fn=query_knowledge_base,
        name="query_knowledge_base",
        description=(
            "在知识库中做一次 BM25+向量混合检索，返回带 [n] 编号的原文片段。"
            "适合回答单个事实问题。可选 file_name 限定文件；跨文件问题不要传 file_name。"
        ),
    ),
    FunctionTool.from_defaults(
        fn=research_topic,
        name="research_topic",
        description=(
            "研究模式：围绕主题自动生成多个检索角度，分别混合检索、去重筛选，"
            "再生成带 [n] 引用的结构化研究报告。适合综述、对比、多角度分析。"
            "不确定资料范围时先用 list_files；跨文件研究不要传 file_name。"
        ),
    ),
    FunctionTool.from_defaults(
        fn=get_current_datetime,
        name="get_current_datetime",
        description="获取当前日期和时间。用户问“今天几号”“现在几点”“今天星期几”时使用。",
    ),
    FunctionTool.from_defaults(
        fn=get_today_weather,
        name="get_today_weather",
        description="获取指定城市今日天气。用户问“XX 今天天气怎么样”时使用，需要传入城市名。",
    ),
]


# ============================================================
# 8) Memory + 组装 Agent
# ============================================================
# Memory 是 llama-index 0.14 的新记忆组件，替代已废弃的 ChatMemoryBuffer。
# token_limit 表示只保留最近约 N 个 token 的历史，防止无限增长。
# 注意：memory 不是 Agent 的构造参数，而是在每次 agent.run() 时传入；
# 复用同一个对象，历史才会跨轮累积。
memory = Memory.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)

# 仍然使用 ReActAgent：OpenAILike 不是 FunctionCallingLLM，
# ReActAgent 用文本式“思考+行动”调用工具，兼容任意 LLM。
agent = ReActAgent(
    tools=tools,
    llm=Settings.llm,
    system_prompt=(
        "你是一个严谨的中文研究助手。回答时请遵守："
        "1) 当用户问“有哪些资料/文件”，或你不确定 data/ 里有什么时，先调用 list_files；"
        "2) 用户要求研究报告、多角度分析、对比或综述时，优先调用 research_topic；"
        "3) 用户只问一个具体事实时，调用 query_knowledge_base；"
        "4) 日期时间调用 get_current_datetime，天气调用 get_today_weather；"
        "5) 跨文件研究不要传 file_name；只有用户明确限定某个文件时才传；"
        "6) 回答知识库内容时必须保留工具返回的 [n] 引用编号，并说明来源文件；"
        "如果工具返回的是研究报告，请原样保留其结构、引用编号和“引用来源”部分；"
        "7) 如果工具输出包含 SIMULATED_RETRIEVAL_FAILURE 或任何错误/异常信息，"
        "先用一句话向用户说明工具出错，然后立即重试同一个工具一次；"
        "重试成功后，如果工具输出里有“【系统说明】……请在最终回答开头用一句话向用户说明”"
        "这样的提示，必须照做，在最终回答开头注明“检索过程中出现一次模拟故障，已重试成功”；"
        "绝不编造结果；"
        "8) 如果工具返回“未检索到”，要说明没有找到，并按提示给出下一步"
        "（list_files、换关键词、去掉 file_name），不要编造；"
        "9) 结合对话历史理解追问；不确定时明确说不确定。"
    ),
    verbose=True,  # 打印每一步的思考与工具调用，方便观察决策过程
)


# ============================================================
# 9) 交互式对话
# ============================================================
async def main() -> None:
    print("\n进入研究模式（输入 exit / quit / 退出 结束）。")
    print("建议按顺序试：")
    print("  1. data/ 里有哪些资料？                      <- list_files")
    print("  2. 星河知识库平台和青禾社区图书馆在故障处理与数据同步上有什么异同？")
    print("     请写一份带引用的研究报告。                  <- research_topic（会触发模拟故障）")
    print("  3. 北京今天天气怎么样？                       <- 天气工具")
    print("  4. 我刚才让你研究的是哪两个平台？             <- 验证 Memory 短期记忆")
    if SIMULATE_RETRIEVAL_FAILURE:
        print(
            "\n[演示提示] 本进程第一次检索会被故意注入一次 "
            "SIMULATED_RETRIEVAL_FAILURE；Agent 会解释错误并重试。"
            "设置 SIMULATE_RETRIEVAL_FAILURE=0 可关闭。"
        )
    print()

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break
        if not user_input:
            continue

        # 把同一个 memory 对象传进去，Agent 才能看到之前几轮的对话。
        events_before = _simulated_failure_events
        try:
            handler = agent.run(
                user_msg=user_input,
                memory=memory,
                max_iterations=12,
            )
            result = await handler  # result 是 AgentOutput
            content = (
                result.response.content
                if result.response
                else "（Agent 未返回内容）"
            )
        except Exception as exc:
            # 即使 Agent 本身出错，交互循环也不退出。
            content = (
                f"抱歉，本轮处理失败：{exc}\n"
                "下一步建议：检查网络和 API Key 后重试，或换一种问法。"
            )

        # 演示兜底：模型有时不会复述工具返回里的“已重试成功”说明，
        # 这里在交互层再明确提示一次，确保用户一定知道发生过模拟故障。
        if (
            _simulated_failure_events > events_before
            and "模拟故障" not in content
        ):
            content = (
                "（说明：本次检索过程中出现一次模拟故障，已自动重试成功。）\n\n"
                + content
            )

        print(f"\n助手：{content}")


if __name__ == "__main__":
    asyncio.run(main())
