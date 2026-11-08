# LlamaIndex + DeepSeek RAG / Agent 示例

从最小 RAG 到带研究模式、长期记忆和重复调用防护的 Agent 示例：用 LlamaIndex 混合检索（BM25 关键词 + Embedding 向量）查找相关片段，交给 DeepSeek 生成带引用的回答。

## 技术栈

- 检索：LlamaIndex **混合检索**——`BM25Retriever`（中文按单字、英文按单词分词）+ `OpenAIEmbedding` 向量语义检索（硅基流动 `BAAI/bge-m3`），由 `QueryFusionRetriever` 按 reciprocal rank 融合
- 生成：DeepSeek（模型名 `deepseek-flash`，接口 `https://api.deepseek.com`）
- 引用：stage3 使用 `CitationQueryEngine` 生成 `[1]`、`[2]` 编号；stage6 的研究报告由程序按 `[n]` 编号自动追加“引用来源”清单
- 持久化：`SimpleVectorStore` 索引落盘到 `storage/`，下次启动直接加载
- 记忆：`Memory` 短期记忆 + `StaticMemoryBlock` 长期偏好块（`priority=0`，永不裁剪）
- 健壮性：工具异常兜底、空结果引导、连续重复调用检测

## 脚本

| 脚本 | 说明 |
| --- | --- |
| `stage2.py` | 基线版：仅 BM25 关键词检索 |
| `stage3.py` | 升级版：Embedding 向量检索 + 混合检索 + 带引用回答 + 索引持久化 |
| `stage4.py` | Agent 版：混合检索 + 引用问答 + 日期/天气工具 + `ChatMemoryBuffer` 短期记忆 |
| `stage5.py` | 研究模式：工具异常健壮性 + `Memory` + `list_files` + 多角度检索报告 + 故意注入一次检索故障 |
| `stage6.py` | 长期记忆 + 重复调用检测：`StaticMemoryBlock` 长期偏好 + 连续重复调用拦截 + `/memory`、`/limit`、`/guard-demo` 交互命令 |

## 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 官方 PyPI 直连较慢，建议用国内镜像（清华源）。若报缺少 venv，先装 `python3-venv`。

## 配置密钥

```bash
cp .env.example .env
```

编辑 `.env`，填入你的密钥：

```
DEEPSEEK_API_KEY=sk-你的 DeepSeek 密钥
DEEPSEEK_MODEL=deepseek-flash

# 向量嵌入（硅基流动，OpenAI 兼容接口）
EMBED_API_KEY=sk-你的硅基流动密钥
EMBED_API_BASE=https://api.siliconflow.cn/v1
EMBED_MODEL=BAAI/bge-m3

# stage5/stage6 演示：进程内第一次检索故意失败一次（1=开启，0=关闭）
SIMULATE_RETRIEVAL_FAILURE=1
```

> 有效模型 id 以 `https://api.deepseek.com/models` 为准；`deepseek-v4-flash` 不是有效 id。
>
> DeepSeek 不提供 embedding 接口，因此 `EMBED_MODEL` 继续使用硅基流动 `BAAI/bge-m3`。
> 另外，`llama-index-embeddings-openai 0.7.0` 的 `OpenAIEmbedding` 必须使用 `model_name=`；
> 传 `model=` 且模型名不在 OpenAI 官方枚举里会抛 `ValueError`。

> `.env` 已被 `.gitignore` 排除，不会提交到 git。

## 运行

```bash
python stage2.py   # 基线版：仅 BM25
python stage3.py   # 升级版：混合检索 + 引用 + 持久化
python stage4.py   # Agent 版：工具调用 + ChatMemoryBuffer
python stage5.py   # 研究模式：Memory + list_files + 多角度检索报告
python stage6.py   # 长期记忆 + 重复调用检测（交互式）
```

`stage6.py` 以交互方式运行，启动后可直接输入问题，也支持 `/help`、`/memory`、
`/limit`、`/guard-demo` 等不调用模型的命令。

`stage3.py` 首次运行会调用 embedding 接口构建索引，并持久化到 `storage/`；再次运行会直接加载，不再重复切分和 embedding。

> 修改 `data/` 中的文档后，索引不会自动更新，需删除 `storage/` 重新构建：
>
> ```bash
> rm -rf storage/
> python stage3.py
> ```

## stage5：研究模式

`stage5.py` 在 stage4 的基础上增加了：

- **工具异常兜底**：工具抛出的异常会被 LlamaIndex Agent 框架转成错误观察值，Agent 不会崩溃，
  而是按系统提示向用户说明故障并重试。
- **空结果引导**：检索不到内容时，工具返回明确的“下一步建议”（先 `list_files`、换关键词、
  去掉 `file_name`），避免模型编造。
- **人为故障演示**：进程内第一次检索会故意抛一次 `SIMULATED_RETRIEVAL_FAILURE`；
  设置 `SIMULATE_RETRIEVAL_FAILURE=0` 可关闭。
- **Memory 短期记忆**：用 `Memory.from_defaults(token_limit=30000)` 替代 `ChatMemoryBuffer`
  （研究报告较长，`token_limit` 太小会把历史整段刷掉导致追问失忆）。
- **`list_files` 工具**：Agent 先查看 `data/` 里的资料清单，再决定检索哪个文件。
- **研究模式 `research_topic`**：主题 → LLM 拆成多个检索角度 → 每个角度混合检索 →
  按 `node_id` 去重并按命中次数筛选 → 生成带 `[n]` 引用的报告 → 程序追加“引用来源”清单。

## stage6：长期记忆与重复调用检测

`stage6.py` 保留 stage5 的研究模式流水线（`research_topic`、模拟故障、空结果引导等），
并新增两块能力：

### 1. 长期记忆 `StaticMemoryBlock`

- 用 `StaticMemoryBlock(name="static_preferences", priority=0)` 保存一份固定偏好
  （简体中文回答、保留 `[n]` 引用、先结论后依据、不编造）。
- `priority=0` 表示永不裁剪：即使通过 `/limit` 把 `token_limit` 调得很小、
  短期历史被归档，静态偏好仍会注入每一轮对话。
- `insert_method=USER`：把偏好前置到最新用户消息，避免 ReAct 下出现两条 system 消息
  （部分 OpenAI 兼容网关只认第一条或直接拒绝）。

### 2. 连续重复调用检测

- 同一用户轮次内，如果 Agent 连续用**完全相同的工具 + 完全相同的参数**调用，
  第二次起不再执行工具，而是返回结构化 JSON 提示：
  `status="duplicate_call_blocked"`、`previous_result_excerpt`、`next_steps`。
- 模型应基于已有结果作答，或修改参数（更具体的关键词、正确的文件名）后重试，
  不得再次提交相同调用。
- 检测状态在每个用户轮次开始时重置，跨轮重复检索是合法的。
- 如果上一次调用抛异常（`status="error"`），允许用相同参数重试，
  这样不会破坏“模拟故障后立即重试”的教学行为。

### 3. 交互命令

以下命令不会调用模型，也不产生 API 费用：

| 命令 | 说明 |
| --- | --- |
| `/help` | 显示可用命令 |
| `/memory` | 查看 `token_limit`、活跃历史、静态偏好是否仍被注入 |
| `/limit <n>` | 修改短期记忆 `token_limit`；`token_flush_size` 会按 10% 同步调整 |
| `/guard-demo` | 确定性演示重复调用拦截（第 2 次相同调用被拦截，换参数正常执行） |

### 4. 推荐验证顺序

```text
data/ 里有哪些资料？                                   # list_files
星河知识库平台和青禾社区图书馆在故障处理与数据同步上有什么异同？请写一份带引用的研究报告。
                                                        # research_topic，触发一次模拟故障并自动重试
北京今天天气怎么样？                                    # get_today_weather
我刚才让你研究的是哪两个平台？                          # 验证短期记忆
/memory                                                 # 查看记忆与静态偏好状态
/limit 500                                              # 收缩短期历史
根据你的长期偏好，回答问题时需要遵守哪些要求？          # 验证静态偏好未被裁剪
/guard-demo                                             # 验证重复调用拦截
```

> 注意：`/limit` 调小后，短期记忆可能只保留最近一个完整轮次，追问较早内容会“失忆”；
> 这是预期行为。静态偏好块不受影响，始终保留。

## 数据

把要检索的文档（`.txt` 等）放进 `data/` 目录即可，脚本会加载、切分并建索引。
所有 stage 共用 `storage/` 索引；`data/` 更新后需删除 `storage/` 重新构建。
