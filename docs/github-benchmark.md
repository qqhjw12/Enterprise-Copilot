# 对标 GitHub 优秀项目：选型、落地与差距

这份文档回答一个问题：**这套代码里的每个关键机制，是从哪个成熟项目学来的，落到哪个文件，量化效果如何，哪些故意没抄。**

选型标准只有三条：

1. 机制能落到具体文件，不是"参考了某某架构"这种空话；
2. 能在零 API Key、零外部服务的环境里验证（否则 CI 不可复现）；
3. 不引入与收益不匹配的依赖（例如为了一个 span 引入整套 OTel SDK）。

---

## 一、总览

| # | 参考项目 | 借鉴的机制 | 本项目落点 | 可验证结果 |
|---|---|---|---|---|
| 1 | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | `interrupt()` + `Command(resume=...)`、checkpointer、`recursion_limit` | `runtime/graph/{builder,nodes,runner,checkpointer}.py` | 中断→杀进程→重启→原参数 resume 成功；图循环硬上限 8 轮 |
| 2 | [langchain-ai/agent-inbox](https://github.com/langchain-ai/agent-inbox) | approve / edit / reject 三态、审批负载带完整参数 | `nodes.approval_node`、`web/app.py` | 拒绝后写库 0 条、模型不再重试（有测试） |
| 3 | [langchain-ai/rag-research-agent-template](https://github.com/langchain-ai/rag-research-agent-template)（已归档） | 混合检索 + 分阶段评估 + 多查询 | `runtime/rag/{pipeline,query_transform}.py`、`eval/` | 改写子集 Recall@5 0.800 → **1.000** |
| 4 | [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | FastMCP、`ToolAnnotations`、structured output | `runtime/mcp_layer/{server,client,catalog}.py` | 5 个工具全部带注解与 outputSchema；契约快照测试 |
| 5 | [langchain-ai/langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) | `MultiServerMCPClient`；以及它"每次调用新建 session"的坑 | `runtime/mcp_layer/client.py` | 单次工具调用 1.5 s → **7–10 ms** |
| 6 | [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | guardrails / tripwire、输入闸门阻塞执行、tool guardrails | `runtime/guardrails/`、`runner.py`、`nodes.tools_node` | 注入请求在模型调用前被拦；10 个用例 |
| 7 | [vercel/ai](https://github.com/vercel/ai) | UI Message Stream v1（`text-delta` / `tool-*` / `data-*` / `[DONE]`） | `runtime/api/ui_stream.py` | 4 个协议测试；前端可直接用 `useChat` |
| 8 | [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | 检索/生成分层评估、context precision/recall | `eval/metrics.py`、`scripts/eval_gate.py` | Recall / MRR / nDCG / MAP / P@5 + Bootstrap CI |
| 9 | [langfuse/langfuse](https://github.com/langfuse/langfuse) | 以 trace 为单位串 agent/llm/tool，GenAI 语义属性 | `runtime/obs/spans.py`、`common/middleware.py` | `trace_id` 贯穿 HTTP → 图 → LLM → 工具 |
| 10 | [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix) | span 级观测、OpenInference/GenAI 属性命名 | 同上 | 每条 span 是可直接入库的 JSON |
| 11 | [BerriAI/litellm](https://github.com/BerriAI/litellm) | 模型目录、按 key 预算、超预算降级/拒绝 | `runtime/llm/{gateway,catalog}.py`、`platform_api` | 70% 降级 cheap，100% 返回 402 |
| 12 | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | 解析→切分→混合召回→重排→带锚点引用；文档生命周期 | `runtime/rag/{parsers,chunkers,pipeline,lifecycle}.py` | 43 chunks / 10 篇；改 1 篇只重建 1 篇（嵌入量 100% → 13.6%） |
| 13 | [HumanLayer/humanlayer](https://github.com/HumanLayer/humanlayer) | 审批是一等状态，不是一次函数调用 | `runner.get_state`、`/threads/{id}` | `awaiting_approval` 可跨进程查询/恢复 |
| 14 | [pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) | 依赖显式注入、类型化工具、可评估 | `GraphDeps`、`ToolSpec`、`eval/` | deps 不进 checkpoint（可序列化约束） |

> 说明：项目 3 已归档（最后推送 2024-12），但它的"混合召回 + 分阶段评估 + Ragas"结构仍然是最清晰的参考；我们只取结构，不追它的依赖版本。

---

## 二、逐个拆：抄了什么、怎么落地、哪里不同

### 1. LangGraph：中断与持久化

**上游做法.** `interrupt(payload)` 在节点内挂起，`Command(resume=value)` 从断点继续；状态由 checkpointer 持久化，`thread_id` 是恢复键。

**本项目落地.**

- `runtime/graph/nodes.py::approval_node` 只做两件事：`interrupt(payload)` 和把决策写回 state。恢复时节点会从头重放，所以这里**不能有副作用**（注释里写死了这条）。
- `runtime/graph/runner.py` 把 `interrupt` 事件与 `done` 合并，解决了一个真实竞态：流式消费在 interrupt 处 `break`，`astream` 生成器被丢弃，`next=("approval",)` 还没落盘，紧接着 resume 就报"没有待审批的操作"。现在必须消费到 `done` 再返回。
- `recursion_limit = max_iterations * 6 + 20` 是图级保险：即使条件边写错，也不会无限循环烧钱。

**哪里不同.** 没抄 Agent Inbox 的完整收件箱（多租户待办列表、批量审批）。我们只实现单会话审批；企业落地时这层通常接 OA/BPM，不适合放在 Agent 平台里重造。

### 2. Agent Inbox：审批的三种动作

**上游做法.** 审批卡片展示 tool name + args，支持 approve / edit / reject / respond；edit 后把修改过的参数送回图。

**本项目落地.** `approval_node` 支持三种动作，其中两个细节来自踩坑：

- **edit 是"补丁"而不是"整包替换"**：只覆盖 `args` 里出现的键，避免前端漏传字段把工号清空。
- **reject 是终止态**：不回到 agent。否则模型看到 `rejected_by_user` 报错，会换个参数再请求一次，等于绕过审批。测试 `test_write_tool_rejected_does_not_execute` 断言"拒绝后模型调用次数仍为 1，写库 0 条"。

**哪里不同.** 上游支持"respond"（用自然语言回复审批人）；我们没有做，因为企业内部审批需要留痕，自由文本不适合作为审批结论。

### 3. RAG Research Agent：混合检索 + 分阶段评估

**上游做法.** `EnsembleRetriever` 融合 BM25 与向量召回；Ragas 做 faithfulness / answer relevancy；按 step 记录评估结果。

**本项目落地.**

- `pipeline.py` 固定顺序：Dense + BM25 → RRF(k=60) → Rerank → 阈值过滤。顺序不可配置——一旦可配置，线上就会出现"某个项目偷偷关掉权限过滤"。
- 权限过滤是**召回前 predicate**，不是取回后再筛；否则 top_k 会被别人的文档挤满。`access.chunk_visible` 做 tenant → classification → allowed_scopes → allowed_roles 四层判断，缺 tenant 一律不可见（fail-closed）。
- 评估脚本从 4 路扩到 5 路（新增 `full_mq`），指标从 Recall/MRR 扩到 nDCG@5、MAP@5、P@5、95% Bootstrap CI。

**多查询的具体实现（`runtime/rag/query_transform.py`）：**

- **受控领域词表**（等价于 Elasticsearch 的 synonym graph）：员工说"内网"，制度写"VPN / 远程接入"；员工说"涨工资"，制度写"调薪 / 普调"。只扩展 BM25 查询，不增加 embedding 调用。
- 词表没命中就原样返回，**不做伪相关反馈**。试过一版 RM3-lite（取首轮 BM25 top-3 文档的高 IDF 词），结果是查询漂移：问"公司有没有太空电梯补贴"，BM25 靠"公司/有没有"误召回年假制度，反馈词全部来自那篇文档，重排覆盖率刷到 1.0，噪声成了满分证据，最终答成"年假 10 天"。这个坑调阈值救不回来，只能靠"扩展词来自受控词表"约束。
- 重排只在**词表命中**（`len(variants) > 1`）时使用"原问题 + 扩展词"；没有命中就只用原问题。这样既能让"内网 → VPN"的目标文档在重排阶段不被打下去，又不会让语料外问题自我强化。

**量化（50 条，43 chunks，离线 hashing + heuristic rerank）：**

| 管线 | Recall@5 | 改写子集 | nDCG@5 | MRR | p50 |
|---|---|---|---|---|---|
| RRF + Rerank | 0.960 | 0.800 | 0.916 | 0.902 | 2.33 ms |
| RRF + Rerank + MQ | **1.000** | **1.000** | **0.930** | 0.907 | 2.39 ms |

MRR 降了 0.007：多召回了两条边缘片段，排序被稀释了一点；换来改写子集 20 个点的 Recall，这个交换是划算的，但不粉饰。

**诚实边界.** 同义词表是在看到两条未命中样本（"内网/VPN"、"涨工资/调薪"）之后补的，因此 1.000 有**评估集过拟合**成分。正确做法是留出集或换语料复测；生产上应该从检索日志里挖"零命中查询"来维护词表，而不是拿评估集补词。本文不把它包装成"泛化提升"。同样，43 chunks 的语料里 Dense 就能到 0.96，混合检索的绝对优势被语料规模掩盖了。

### 4. MCP Python SDK：工具注解与结构化输出

**上游做法.** FastMCP 从函数签名生成 inputSchema；`ToolAnnotations` 提供 `readOnlyHint / destructiveHint / idempotentHint / openWorldHint`；返回 Pydantic 模型时生成 `outputSchema` 与 `structuredContent`。

**本项目落地.**

- `server.py` 的 5 个工具全部带注解：3 个只读、2 个写操作；`destructiveHint=False` 是显式写的——不写时协议默认为 true，客户端会把"提交请假"当成删除级危险操作。
- `idempotentHint=True` 不是嘴上说说：写操作由客户端注入 `idempotency_key`，`backends` 按 key 去重，重复提交返回 `deduplicated=True`。
- `catalog.py` 把工具契约输出成可 diff 的 JSON：`schema_hash = sha256(input + output)[:16]`，`contract_version` 单独版本化。
- `tests/test_tool_contract.py` 做快照比对，改参数必须显式更新快照；同时断言 `caller` 绝不出现在模型可见 schema 里（caller 由运行时注入）。

**哪里不同.** 没做 `tools/list_changed` 通知与工具热更新。企业内部工具变更走发布流程，热更新反而让"模型看到的工具"和"审计记录"对不上。

### 5. MCP Adapters：一个真实性能坑

**上游做法.** `MultiServerMCPClient` 把 MCP 工具转成 LangChain Tool，`tool.ainvoke()` 用起来最顺手。

**踩到的坑.** `tool.ainvoke()` 每次新建 `ClientSession`；stdio 传输下等于**每次工具调用重新拉起一个 Python 进程**。实测 1.4–1.5 s/次，全部耗在解释器启动。

**本项目落地.** `client.McpToolbox` 自己持一条长连接 session，断线重建；单次调用降到 7–10 ms（约 200×）。这不是"优化了一下"，是把不可用的东西变成可用。

### 6. OpenAI Agents SDK：Guardrails

**上游做法.** `@input_guardrail` / `@output_guardrail` 返回 `GuardrailFunctionOutput(tripwire_triggered=...)`；输入闸门默认并行，可切阻塞；工具级 guardrail 在执行前后检查参数/结果。

**本项目落地（`runtime/guardrails/core.py`）.**

| 层 | 检查 | 触发后 |
|---|---|---|
| 输入 | 覆盖指令、套取系统提示、越狱、绕过审批、密钥、超长 | 阻塞，**不调用模型**，返回固定安全话术 |
| 输入 | 手机号 / 身份证 / 银行卡 | 脱敏后继续（用户可能只是在陈述事实） |
| 输出（流式） | 风险前缀（`ZQ-`/`sk-`/`AKIA`/系统提示）出现即扣住 token | 最多晚 8 字符下发；流结束整体校验 |
| 输出（收尾） | 跨租户金丝雀、密钥、系统提示特征串 | 替换为拦截话术，`status=blocked` |
| 工具参数 | 模型自带 `caller`、参数夹带密钥 | 不执行工具，返回 `guardrail_blocked` |

三个工程决策：

1. **阻塞而非并行**：并行延迟低，但模型可能已经花了 token、甚至执行了工具（上游文档也明确提示这个取舍）。企业场景里"没花钱"比"快 300 ms"重要。
2. **判定结果不回传原文**：`Verdict.as_event()` 只给规则名与计数。否则日志本身成了密钥泄露渠道。
3. **输出闸门不是权限过滤的替代品**：租户隔离在召回前就做了；输出闸门用来兜住"过滤被绕过"的意外，并让它变成一个可告警的 `guardrail` 事件。

**哪里不同.** 上游可以用一个便宜的 LLM 做 guardrail agent；我们离线环境没有模型，用的是确定性规则。接真实模型后可以把"语义注入检测"加在这一层，接口不变。

### 7. Vercel AI SDK：UI Message Stream

**上游协议.** 响应头 `x-vercel-ai-ui-message-stream: v1`；`data: {"type":"text-delta","id":...,"delta":...}`；工具用 `tool-input-available` / `tool-output-available`；审批用 `tool-approval-request`；结束 `data: [DONE]`。

**本项目落地（`runtime/api/ui_stream.py`）.**

- 内部事件（`route/retrieval/tool_call/interrupt`）→ 标准 part：
  - `token` → `text-start` / `text-delta` / `text-end`
  - `tool_call` → `tool-input-available`，写操作额外发 `tool-approval-request`
  - `tool_result` → `tool-output-available`（按工具名队列匹配 call id）
  - `citations` / `guardrail` / `approval` → `data-citations` / `data-guardrail` / `data-approval`
- 新增 `POST /v1/chat/ui-stream` 与 `POST /v1/threads/{id}/resume/ui-stream`，保留原 `/chat/stream` 的领域事件协议（内部排障更好读）。

**为什么值得做.** 前端不用为"某个 Agent 平台"写适配层；`useChat` / assistant-ui 可以直接消费。协议测试断言的是完整 part 序列，不是"响应里有字"。

### 8. Ragas：为什么没有直接用它

**上游做法.** faithfulness / answer_relevancy / context_precision / context_recall，核心依赖 LLM 裁判。

**本项目的取舍.**

- 检索层评估必须**确定性、可复现、零成本**，否则 CI 门禁会被模型波动搞成随机失败。
- 我们实现 Recall/MRR/nDCG/MAP/P@5；在"每条查询只有一个相关文档"的标注下，Ragas 的 context_precision/recall 退化为 P@5 与 Recall@5，二者可对齐。
- 生成层的 faithfulness 没有做——没有 Key 时"生成"是抽取式的，用 LLM 判分没有意义。接真实模型后应补：抽样 100 条，跑 `ragas.evaluate(..., metrics=[faithfulness, answer_relevancy])`，只作为夜间任务，不进阻塞式 CI。

### 9. Langfuse / Phoenix：span 而不是日志

**上游做法.** 以 trace 为单位组织 generation/span；属性命名遵循 OpenTelemetry GenAI semconv；Langfuse 提供 OTLP 入口，Phoenix 用 OpenInference。

**本项目落地（`runtime/obs/spans.py`）.**

- 自实现 60 行 span：`trace_id / span_id / parent_span_id / name / duration_ms / status / attributes`，一条 JSON 日志一行。
- 属性名直接采用 semconv：`gen_ai.operation.name`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.tool.name`、`gen_ai.tool.call.id`、`gen_ai.conversation.id`。
- `trace_id` 由中间件从 W3C `traceparent` 或 `x-trace-id` 复用，响应头回写 `x-trace-id`，SSE `start/done` 事件里也带。
- span 覆盖三层：`invoke_agent`（runner）→ `chat`（gateway）→ `execute_tool`（tools_node）。

**哪里不同.** 没引 `opentelemetry-sdk`（会带来 exporter/采样器/资源探测一整套依赖）。输出格式已对齐，接 collector 时由 sidecar 转换即可；这是"应用侧不绑死某个观测厂商"的主动选择。

### 10. LiteLLM：预算与降级

**上游做法.** 虚拟 key + 每分钟/每月预算；超预算拒绝；支持 fallback 模型。

**本项目落地.** `BudgetGuard` 逐次记账，70% 降级 cheap 模型，100% 抛 `BudgetExceeded` → HTTP 402；模型目录由控制面下发（带 TTL + 负缓存），执行面不硬编码价格。

**哪里不同.** 没做跨进程预算聚合（Redis）。当前预算是 per-run，适合"一次对话不许烧穿"；企业级配额（按人/按月）需要把 `spent_usd` 落到 Redis/PG，这是下一步而不是已完成。

### 11. RAGFlow：解析与引用

**上游做法.** DeepDoc 做版面分析/OCR/表格抽取；chunk template 按文档类型切；引用能定位到页/块。更关键的是**文档生命周期**：每份文档有解析状态、版本、增删改都要反映到 chunk 上，而不是每次都重建整个索引。

**本项目落地.**

- 解析：PDF(pypdf)/MD(front-matter)/HTML(BeautifulSoup)；`SemanticChunker` 与 `RecursiveChunker` 可切换；引用带 `chunk_id/doc_id/title/section/snippet/score`。
- 生命周期：`runtime/rag/lifecycle.py` 用**双指纹**判定变更，`pipeline.sync_documents()` 据此只重建真正变了的文档：

  | 指纹 | 覆盖范围 | 变了意味着什么 |
  |---|---|---|
  | `content_hash` | 仅正文 | 要重新切分 + 重新嵌入 |
  | `policy_hash` | `tenant_id` / `classification` / `allowed_scopes` / `allowed_roles` | 向量可复用，但元数据必须重写 |

  两个指纹刻意分开。只对正文哈希的后果是：某文档从 `internal` 上调为 `confidential`，正文一字未改 → 判定 unchanged → 整体跳过 → 库里留着旧密级，普通员工照样检索得到。**这是一次不报错、不告警的权限降级**，只会在合规检查时被发现。所以 `updated_policy` 是独立的变更类型，不是 `unchanged` 的一种。

- 清单：`index_manifest.json` 记 `doc_id → {content_hash, policy_hash, chunks, updated_at}`。删掉它等于宣告"不知道库里有什么"，下次全量重建（安全，只是慢）。
- 下架：`prune=True` 时源文件消失的文档连 chunk 一起删。两条召回通道（向量库 + BM25）必须同时删，否则一边还在召回废止条款。

**量化（10 篇 / 44 chunks，真实 CLI 跑出来的）.**

| 场景 | 重建文档 | 重建 chunk | 嵌入耗时 |
|---|---|---|---|
| 首次全量 | 10 | 44 | 0.006 s |
| 原样重跑 | 0 | 0 | 0.000 s |
| 改 1 篇正文 | 1 | 6（旧 6 个先删） | 0.001 s |
| 只改 1 篇密级 | 1 | 6 | 0.001 s |
| 删 1 篇 + `--prune` | 0 | 删除 4 | — |

**哪里不同.** 没有 OCR、版面分析、表格结构化，扫描件 PDF 只能拿到空文本——这是明确的缺口，不假装支持。也没有 RAGFlow 的解析任务队列与解析状态机；我们用"指纹 + 清单"这种更轻的方式，好处是不引入任务系统也能正确增量，代价是不支持"解析失败重试"这类运营语义（切分失败的文档会被排除在清单外，下一轮重新尝试，等价于无限重试）。


### 12. Pydantic AI / HumanLayer：类型化依赖与审批状态

**上游做法.** Pydantic AI 用类型化 deps + 结构化输出 + evals；HumanLayer 把审批做成可路由、可升级、可审计的状态机。

**本项目落地.** `GraphDeps` 显式注入（LLM 网关、工具箱、caller、settings），**不进 state/checkpoint**——state 里只放可序列化数据，这是上 PostgresSaver 时最容易踩的坑。审批状态通过 `get_state` 暴露为 `awaiting_approval`，可跨进程查询。

**哪里不同.** 审批没有接 Slack/邮件/工单系统，也没有超时升级（SLA 到点自动转交）。企业落地时这层应接 BPM，而不是在 Agent 里写死。

---

## 三、本轮具体改动（可逐条 diff）

| 文件 | 改了什么 | 验证 |
|---|---|---|
| `runtime/guardrails/{__init__,core}.py` | 新增输入/输出/工具参数三层 guardrail，tripwire 语义 | `tests/test_guardrails.py`（14 条） |
| `runtime/graph/runner.py` | 输入闸门前置、`StreamGuard` 逐 token 过闸、收尾整体校验、`trace_id`、span | 拦截用例断言 `spent_usd=0`、checkpoint 不存在 |
| `runtime/graph/nodes.py` | 工具执行前参数校验 + `execute_tool` span | 越权/注入不执行 |
| `runtime/rag/query_transform.py` | 受控领域词表扩展；拒绝伪相关反馈（查询漂移）；只扩展 BM25 | `tests/test_query_transform.py` |
| `runtime/rag/pipeline.py` | 多查询 RRF、扩展词重排、trace.variants | 评估第 5 路 |
| `eval/metrics.py` | Recall/MRR/nDCG/MAP/P@5 + Bootstrap CI | 指标单测 + 评估报告 |
| `eval/run_eval.py` | 五路对比、CI、nDCG、P@5 | `var/eval_report.json` |
| `runtime/mcp_layer/{client,server,catalog}.py` | ToolAnnotations、outputSchema、schema_hash、契约版本 | `tests/test_tool_contract.py` + 快照 |
| `runtime/api/{ui_stream,tools}.py` | AI SDK UI Message Stream、`GET /v1/tools` | `tests/test_ui_stream.py` |
| `runtime/obs/spans.py`、`common/middleware.py` | GenAI semconv span、`trace_id` 贯穿 | `tests/test_observability.py` |
| `runtime/rag/lifecycle.py` | 双指纹（正文/权限）变更判定、清单持久化、`SyncReport` | `tests/test_incremental_index.py`（16 条） |
| `runtime/rag/pipeline.py` | `sync_documents/sync_directory`、状态丢失退化为全量、两条召回通道同时增删 | 改 1 篇只嵌入 6/44 chunk |
| `runtime/rag/ingest.py` | 默认增量，新增 `--prune/--force/--dry-run/--full` | CLI 六场景实测 |
| `.github/workflows/ci.yml`、`scripts/eval_gate.py` | lint + 双版本测试 + 检索门禁 | 本地 `ruff check .` 0 错、gate PASS |
| `pyproject.toml`、`.pre-commit-config.yaml` | ruff 规则（忽略 FastAPI 的 B008） | — |

**数字变化：**

- 测试：51 → 100（新增 49 条，全部零 Key、零外部服务）。
- 检索：Recall@5 0.960 → 1.000；改写子集 0.800 → 1.000；nDCG@5 0.916 → 0.930；代价 +0.06 ms p50。
- 增量：改 1 篇的嵌入量从 100%（44 chunk）降到 13.6%（6 chunk）；无改动时嵌入耗时为 0。
- 静态检查：ruff 76 个问题 → 0。
- 工具契约：5 个工具全部有注解 + outputSchema + 16 位 schema_hash。

---

## 四、仍然没做的（按优先级）

| 优先级 | 缺口 | 参考实现 | 为什么现在不做 |
|---|---|---|---|
| P0 | Postgres checkpointer 的集成测试 | LangGraph `PostgresSaver` + testcontainers | 本机无 Docker；单测覆盖了 sqlite/memory，PG 只在 compose 里声明 |
| P0 | 真实 OTLP 导出 | Langfuse OTLP / Phoenix | 需要 collector，CI 里无法验证；先保证字段对齐 |
| P0 | OIDC / SSO | 企业 IdP | 当前是本地账号 + JWT，够演示治理边界，不够生产 |
| P1 | 增量索引的**向量复用** | RAGFlow 的 parser 缓存 | 权限变更时会重新嵌入同一段正文（正文没变，向量必然一样）。要先让 `VectorStore` 协议暴露 `get_vectors(ids)`，收益只在"频繁改密级"的场景里成立 |
| P1 | CrossEncoder 的真实评估 | `BAAI/bge-reranker-base` | 需要下载权重；接口已留，阈值需重新标定 |
| P1 | 跨进程预算/限流（Redis） | LiteLLM virtual key | 当前 per-run 预算已能防"单次烧穿" |
| P2 | 多 Agent / supervisor | LangGraph multi-agent、Open Deep Research | 单 Agent + 工具循环已覆盖当前业务；多 Agent 的失败模式更复杂 |
| P2 | LLM-as-judge 的生成质量评估 | Ragas faithfulness | 离线无模型；应作为夜间任务而非 CI 门禁 |
| P2 | 审批超时升级 / IM 通知 | HumanLayer | 需要接企业 IM 与组织架构，属于集成工作 |

---

## 五、复现

```bash
python -m pytest -q                 # 100 passed
python -m eval.run_eval             # 五路检索对比
python scripts/eval_gate.py         # Recall / MRR 门禁
python -m ruff check .              # 0 error

# 起服务后验证新协议
curl -N -X POST http://127.0.0.1:8001/v1/chat/ui-stream \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message":"年假有几天"}'
curl http://127.0.0.1:8001/v1/tools -H "Authorization: Bearer $TOKEN"
```
