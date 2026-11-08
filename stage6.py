"""LlamaIndex + DeepSeek：在 stage5（研究模式）基础上增加长期记忆与重复调用检测。

相对 stage5 的两个变化：
  1) Memory 增加 StaticMemoryBlock 长期记忆块（priority=0，永不裁剪）：
     即使把 token_limit 调小、短期历史被归档，静态偏好仍会注入每一轮对话。
  2) 工具增加“连续重复调用”检测：
     同一轮内连续用完全相同参数调用同一工具时，不再静默执行，
     而是返回结构化提示（duplicate_call_blocked），让模型基于已有结果作答或换策略。

stage5 的研究模式流水线（research_topic 工具）保持不变：
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
  7) 工具封装 + 重复调用检测
  8) Memory（StaticMemoryBlock）+ ReAct Agent
  9) 交互式对话（含 /limit、/memory、/guard-demo 命令）

密钥通过 .env 读取，不硬编码在代码里。
"""
import asyncio
import functools
import inspect
import json
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
from llama_index.core.memory import (                            # 新版记忆组件
    InsertMethod,
    Memory,
    StaticMemoryBlock,     # 长期记忆块：内容固定、永不裁剪
)
from llama_index.core.node_parser import SentenceSplitter        # 按句子/长度切分文档
from llama_index.core.retrievers import QueryFusionRetriever     # 融合多个检索器的结果
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES  # 融合算法枚举
from llama_index.core.schema import NodeWithScore                # 带分数的检索片段
from llama_index.core.storage.chat_store.base_db import MessageStatus  # 查询活跃历史
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
# 可通过环境变量或交互命令 /limit 调整；长期偏好不受它影响。
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "30000"))
# 驱逐历史时每次最多刷掉多少 token（比例于 token_limit）。
# 注意：运行时修改 token_limit 时必须同步修改 token_flush_size，
# 否则默认的 3000 会把小限额下的历史一次性冲光。
MEMORY_FLUSH_RATIO = 0.1

# 长期记忆块：内容固定，priority=0 表示永不裁剪。
STATIC_BLOCK_NAME = "static_preferences"
STATIC_PREFERENCES = """【长期偏好｜对所有轮次持续有效】
1. 始终使用简体中文回答。
2. 引用知识库事实时必须保留工具返回的 [n] 编号，并注明来源文件名。
3. 先给结论，再给依据；信息不足时明确说明，不得编造。
4. 如果用户询问你的偏好或回答要求，请复述以上要点。"""

# 重复调用提示里保留多少字符的“上一次结果”，方便模型直接作答。
GUARD_EXCERPT_CHARS = 400

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
            f"下一步建议：删除 {PERSIST_DIR}/ 后重新运行 stage6.py 重建索引，再重试。"
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
            "SIMULATED_RETRIEVAL_FAILURE（stage6 演示）：检索服务暂时不可用，"
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
        print(f"[stage6] 子问题生成失败，改用固定角度：{exc}")
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
        print(f"[stage6] 报告合成失败，返回原始资料：{exc}")
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


# ============================================================
# 7.3) 重复调用检测
# ============================================================
# 目标：同一轮对话内，如果 Agent 连续用“完全相同的工具 + 完全相同的参数”调用，
# 第二次及以后不再执行工具，而是返回结构化提示，让模型基于已有结果作答或换策略。
#
# 注意：
#   - 状态是模块级的，且同步工具函数会在线程池中执行，所以用 threading.Lock；
#   - 每个用户轮次开始时调用 _reset_repeat_guard()，跨轮重复调用是合法的；
#   - 上一次调用如果抛异常（status="error"），允许模型用相同参数重试，
#     否则会破坏 stage5 “模拟故障后立即重试”的教学行为。
_guard_lock = threading.Lock()
_guard_state = {
    "signature": None,   # (tool_name, canonical_args_json)
    "status": None,      # None | "running" | "ok" | "error"
    "output_excerpt": "",
    "repeat_count": 0,
}


def _normalize_arg(value):
    """让等价参数得到相同签名：字符串折叠空白，递归处理容器。"""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (list, tuple)):
        return [_normalize_arg(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_arg(item) for key, item in value.items()}
    return value


def _canonical_args(fn, args, kwargs) -> str:
    """按函数签名绑定参数并补默认值，生成稳定的 JSON 字符串。"""
    sig = inspect.signature(fn)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        payload = dict(bound.arguments)
        # None 与“默认值为字符串”的参数等价：file_name=None -> file_name=""
        for name, param in sig.parameters.items():
            if (
                name in payload
                and payload[name] is None
                and isinstance(param.default, str)
            ):
                payload[name] = param.default
        payload = _normalize_arg(payload)
    except TypeError:
        # 理论上不会发生（调用方已按签名传参）；兜底保证提示仍可生成。
        payload = {
            "args": [_normalize_arg(repr(value)) for value in args],
            "kwargs": {
                key: _normalize_arg(repr(value))
                for key, value in sorted(kwargs.items())
            },
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _reset_repeat_guard() -> None:
    """重置检测状态。每个真实用户轮次开始时调用一次。"""
    with _guard_lock:
        _guard_state.update(
            signature=None,
            status=None,
            output_excerpt="",
            repeat_count=0,
        )


def _duplicate_call_notice(tool_name: str, arguments: dict, repeat_count: int) -> str:
    """构造返回给模型的结构化提示（JSON 字符串，会作为 observation 被看到）。"""
    hint = {
        "status": "duplicate_call_blocked",
        "tool": tool_name,
        "arguments": arguments,
        "repeat_count": repeat_count,
        "previous_result_excerpt": _guard_state["output_excerpt"],
        "message": "检测到连续重复调用同一工具且参数完全相同，本次没有再次执行。",
        "next_steps": [
            "直接使用 previous_result_excerpt 或上下文中已有的工具结果回答用户；",
            "如果信息不足，请修改参数（更具体的关键词或正确的文件名）后重试；",
            "如果该工具无法回答，请换用其他工具或如实说明信息不足；",
            "不要再次提交完全相同的调用。",
        ],
    }
    return json.dumps(hint, ensure_ascii=False, indent=2)


def _guard_repeated_calls(tool_name: str, fn):
    """包装工具函数：连续相同调用不执行，返回结构化提示。

    @functools.wraps 必须保留：FunctionTool 通过 inspect.signature(fn) 生成参数 schema，
    而 inspect.signature 会沿 __wrapped__ 找到原函数。
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        canonical = _canonical_args(fn, args, kwargs)
        signature = (tool_name, canonical)

        with _guard_lock:
            same_call = _guard_state["signature"] == signature
            previous_status = _guard_state["status"]
            if same_call and previous_status in {"ok", "running"}:
                # 命中重复：不执行工具，只累加重复次数并返回提示。
                _guard_state["repeat_count"] += 1
                return _duplicate_call_notice(
                    tool_name,
                    json.loads(canonical),
                    _guard_state["repeat_count"],
                )

            # 标记为 running 后再释放锁，避免并发情况下同一调用被执行两次。
            _guard_state.update(
                signature=signature,
                status="running",
                output_excerpt="",
                repeat_count=0,
            )

        # 执行工具时不持锁（检索可能较慢）。
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            with _guard_lock:
                _guard_state.update(
                    signature=signature,
                    status="error",
                    output_excerpt=_snippet(str(exc), GUARD_EXCERPT_CHARS),
                    repeat_count=0,
                )
            raise  # 交给框架转成错误观察值，并允许模型用相同参数重试

        with _guard_lock:
            _guard_state.update(
                signature=signature,
                status="ok",
                output_excerpt=_snippet(str(result), GUARD_EXCERPT_CHARS),
                repeat_count=0,
            )
        return result

    return wrapper


guarded_list_files = _guard_repeated_calls("list_files", list_files)
guarded_query_knowledge_base = _guard_repeated_calls(
    "query_knowledge_base", query_knowledge_base
)
guarded_research_topic = _guard_repeated_calls("research_topic", research_topic)
guarded_get_current_datetime = _guard_repeated_calls(
    "get_current_datetime", get_current_datetime
)
guarded_get_today_weather = _guard_repeated_calls(
    "get_today_weather", get_today_weather
)


def _run_guard_demo() -> None:
    """确定性演示重复调用拦截：不调用模型，不产生 API 费用。"""
    _reset_repeat_guard()
    first = guarded_get_today_weather("北京")
    second = guarded_get_today_weather("北京")
    third = guarded_get_today_weather("上海")
    print("\n[重复调用检测演示]")
    print("1) 第 1 次调用（正常执行）：", first)
    print("2) 第 2 次相同调用（应被拦截）：\n" + second)
    print("3) 参数不同（应正常执行）：", third)
    _reset_repeat_guard()


tools = [
    FunctionTool.from_defaults(
        fn=guarded_list_files,
        name="list_files",
        description=(
            "列出 data/ 中所有可检索资料文件（文件名、大小、索引片段数）。"
            "当用户问“有哪些资料/文件”，或你不确定该检索哪个文件时，先调用本工具。"
        ),
    ),
    FunctionTool.from_defaults(
        fn=guarded_query_knowledge_base,
        name="query_knowledge_base",
        description=(
            "在知识库中做一次 BM25+向量混合检索，返回带 [n] 编号的原文片段。"
            "适合回答单个事实问题。可选 file_name 限定文件；跨文件问题不要传 file_name。"
        ),
    ),
    FunctionTool.from_defaults(
        fn=guarded_research_topic,
        name="research_topic",
        description=(
            "研究模式：围绕主题自动生成多个检索角度，分别混合检索、去重筛选，"
            "再生成带 [n] 引用的结构化研究报告。适合综述、对比、多角度分析。"
            "不确定资料范围时先用 list_files；跨文件研究不要传 file_name。"
        ),
    ),
    FunctionTool.from_defaults(
        fn=guarded_get_current_datetime,
        name="get_current_datetime",
        description="获取当前日期和时间。用户问“今天几号”“现在几点”“今天星期几”时使用。",
    ),
    FunctionTool.from_defaults(
        fn=guarded_get_today_weather,
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
#
# StaticMemoryBlock 是长期记忆块：
#   - priority=0：llama-index 约定 0 表示“永不裁剪”，
#     即使 token_limit 很小、短期历史被归档，它也始终注入；
#   - static_content 内容固定，_aput 是 no-op，不会被短期消息改写；
#   - insert_method=USER：把静态内容前置到最新用户消息。
#     若用默认的 SYSTEM，ReActAgent 下模型会同时收到 ReAct header 和静态块两条 system 消息，
#     部分 OpenAI 兼容网关只认第一条或直接拒绝；USER 模式更稳妥。
memory = Memory.from_defaults(
    token_limit=MEMORY_TOKEN_LIMIT,
    token_flush_size=max(1, int(MEMORY_TOKEN_LIMIT * MEMORY_FLUSH_RATIO)),
    memory_blocks=[
        StaticMemoryBlock(
            name=STATIC_BLOCK_NAME,
            static_content=STATIC_PREFERENCES,
            priority=0,
            accept_short_term_memory=False,
        )
    ],
    insert_method=InsertMethod.USER,
)

# 仍然使用 ReActAgent：OpenAILike 不是 FunctionCallingLLM，
# ReActAgent 用文本式“思考+行动”调用工具，兼容任意 LLM。
agent = ReActAgent(
    tools=tools,
    llm=Settings.llm,
    system_prompt=(
        "你是一个严谨的中文研究助手。你会收到 <memory><static_preferences> 中的长期偏好，"
        "它们对所有轮次持续有效，必须遵守。回答时请遵守："
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
        "9) 结合对话历史理解追问；不确定时明确说不确定；"
        "10) 如果工具返回 status=duplicate_call_blocked，表示你连续用相同参数重复调用了"
        "同一工具，该调用已被拦截：不要再次提交完全相同的调用；优先使用 "
        "previous_result_excerpt 或对话中已有的工具结果直接回答；信息不足时修改参数"
        "（更具体的关键词、正确的文件名）或换用其他工具；仍无法回答就如实说明。"
    ),
    verbose=True,  # 打印每一步的思考与工具调用，方便观察决策过程
)


# ============================================================
# 8.5) 交互命令：/help /memory /limit /guard-demo
# ============================================================
async def _print_memory_status() -> None:
    """打印 token 限额、活跃历史、静态偏好块状态。"""
    active = await memory.aget_all(status=MessageStatus.ACTIVE)
    active_tokens = sum(memory._estimate_token_count(message) for message in active)
    threshold = int(memory.token_limit * memory.chat_history_token_ratio)
    print(
        f"token_limit={memory.token_limit}  "
        f"token_flush_size={memory.token_flush_size}  "
        f"chat_history_token_ratio={memory.chat_history_token_ratio}  "
        f"驱逐阈值={threshold} tokens"
    )
    print(
        f"短期记忆：{len(active)} 条活跃消息，约 {active_tokens} tokens"
        "（已归档消息不计入；Memory 至少保留最近一个完整对话轮次，"
        "因此可能高于驱逐阈值）"
    )

    for block in memory.memory_blocks:
        if isinstance(block, StaticMemoryBlock):
            tokens = memory._estimate_token_count(block.static_content)
            print(
                f"静态偏好块：name={block.name}  priority={block.priority}  "
                f"约 {tokens} tokens（priority=0，永不裁剪）"
            )

    # aget() 只读取并注入静态块，不写入 chat store，可以安全地用来验证。
    probe = await memory.aget()
    carried = any(
        STATIC_BLOCK_NAME in (message.content or "") for message in probe
    )
    print(f"当前 aget() 是否携带静态偏好：{carried}")
    print("静态偏好内容：\n" + STATIC_PREFERENCES)


async def _set_memory_token_limit(new_limit: int) -> None:
    """修改 token_limit，并立即触发一次惰性驱逐（演示用）。"""
    new_limit = max(1, new_limit)
    old_limit = memory.token_limit
    memory.token_limit = new_limit
    # 必须同步 flush size：否则默认 3000 会把小限额下的历史一次性冲光。
    memory.token_flush_size = max(1, int(new_limit * MEMORY_FLUSH_RATIO))

    forced = False
    try:
        # 私有 API：Memory 平时只在 aput 时驱逐，这里为了演示立即执行。
        await memory._manage_queue()
        forced = True
    except Exception as exc:  # 库版本变化时不影响主流程
        print(f"（提示：立即驱逐失败，将在下一条用户消息时惰性执行：{exc}）")

    print(
        f"token_limit：{old_limit} -> {new_limit}；"
        f"token_flush_size -> {memory.token_flush_size}"
    )
    print(f"驱逐阈值 = {int(new_limit * memory.chat_history_token_ratio)} tokens")
    print(
        "静态偏好块 priority=0，永不裁剪；"
        + (
            "已立即执行一次历史驱逐检查（至少保留最近一个完整轮次）。"
            if forced
            else "将在下一条用户消息时惰性驱逐。"
        )
    )
    await _print_memory_status()


async def _handle_command(command: str) -> None:
    """处理以 / 开头的交互命令。"""
    parts = command.split()
    cmd = parts[0].lower()

    if cmd == "/help":
        print("可用命令：")
        print("  /help            显示本帮助")
        print("  /memory          查看 token_limit、活跃历史、静态偏好是否仍被注入")
        print("  /limit <n>       修改短期记忆 token_limit（静态偏好始终保留）")
        print("  /guard-demo      确定性演示重复调用拦截（不调用模型、不产生费用）")
    elif cmd == "/memory":
        await _print_memory_status()
    elif cmd == "/limit":
        if len(parts) != 2 or not parts[1].isdigit():
            print("用法：/limit <正整数>，例如 /limit 500")
            return
        new_limit = int(parts[1])
        if new_limit < 50:
            print("提示：token_limit 过小会几乎清空短期记忆，但静态偏好仍会保留。")
        await _set_memory_token_limit(new_limit)
    elif cmd == "/guard-demo":
        _run_guard_demo()
    else:
        print(f"未知命令：{parts[0]}。输入 /help 查看可用命令。")


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
    print("\n可用命令（不会调用模型）：")
    print("  /memory     查看记忆状态与静态偏好")
    print("  /limit 500  收缩短期记忆上限，静态偏好不会丢失")
    print("  /guard-demo 确定性演示重复调用拦截")
    print("  /help       查看全部命令")
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

        # 交互命令：不进入 Agent，不消耗模型调用。
        if user_input.startswith("/"):
            await _handle_command(user_input)
            continue

        # 每个真实用户轮次重置重复调用检测：
        # 同一轮内连续相同调用才会被拦截，跨轮重复检索是合法的。
        _reset_repeat_guard()

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
