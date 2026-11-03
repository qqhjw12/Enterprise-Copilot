# Enterprise Copilot

一个**企业级 AI Agent 平台底座**：把「平台治理层」与「Agent 运行时层」彻底拆开，
运行时用 LangGraph 状态图编排（条件路由 / 工具循环 / HITL 中断 / 兜底降级），
工具用 MCP 标准化接入内部系统，知识走 Dense + BM25 + RRF + Rerank 的混合检索，
对外是 SSE 事件流（领域事件 + AI SDK UI 流两种编码），一条 `docker compose up` 起全栈。

不是"聊天框 + 一个知识库"的 Demo。三条硬线贯穿全仓：

| 企业级特征 | 在代码里的落点 |
|---|---|
| 平台治理与执行分离 | `platform_api/`（认证、租户、模型 Catalog、审计）与 `runtime/`（图、MCP、RAG）是两个独立进程、两套依赖，只靠 JWT + 内部 HTTP 通信 |
| 工具接入标准化 | `runtime/mcp_layer/`：FastMCP Server + MultiServerMCPClient，Schema 校验、scope 鉴权、熔断/超时/重试都在协议层 |
| 可观测与可评估 | JSON 结构化日志（request_id / user_id / thread_id / trace_id）、GenAI semconv span、Prometheus 指标、LangSmith/Langfuse 挂钩、per-run 预算、50 条评估集 + CI 门禁 |

**这一版重点补的不是功能数量，而是"企业里真的会卡住上线"的四件事**：

- **安全闸门**：输入/输出/工具参数三层 guardrail（提示注入、密钥、跨租户金丝雀），接入 `runtime/guardrails/`，注入请求在模型调用前就被拦掉，`spent_usd=0`。
- **协议兼容**：新增 `/v1/chat/ui-stream`，输出 Vercel AI SDK 的 UI Message Stream v1，`useChat` / assistant-ui 可直接消费，不必为这个平台写渲染适配层。
- **工具契约**：5 个 MCP 工具补齐 `ToolAnnotations` + `outputSchema` + `schema_hash`，`GET /v1/tools` 可查，快照测试防止"改了参数没通知调用方"。
- **检索可解释**：新增第 5 条管线（多查询改写），改写子集 Recall@5 从 0.800 提到 1.000；评估指标补 nDCG@5 / MAP@5 / P@5 / 95% CI，进 CI 门禁。

每个机制对应哪个 GitHub 项目、抄了什么、哪里故意没抄，见 **[docs/github-benchmark.md](docs/github-benchmark.md)**（14 个项目逐条对照）。

---

## 一、这套东西跑起来是什么样

四条路径，四条不同的代码分支，不是同一个线性流程换皮：

```
用户：年假有几天？                → knowledge 子图：检索 → 有依据才回答 → 带引用
用户：帮我查下我的资产            → 只读工具：agent → tools → agent（本轮循环）
用户：帮我提交年假申请 11-10 到 11-12 → 写操作：先 interrupt 等人批，批了才落库
用户：帮我查下 E2048 的资产        → 越权：MCP Server 返回结构化 permission_denied，不是裸异常
```

`python scripts/smoke.py` 会把上面这 8 个场景一次跑完并打表（当前 8/8 通过）：

```
场景             结果    细节
------------------------------------------------------------------------------------------------
知识问答（员工）       PASS    3 条引用，top1=员工年假与请假制度(0.8202)
语料外问题          PASS    知识库中未收录该信息，建议联系对应部门或在 IT 服务台提交咨询工单。
权限隔离（机密文档）     PASS    员工可见=['员工年假与请假制度', '差旅与费用报销标准', '采购与合同审批权限'] / HR 额外可见=['薪酬与福利管理细则']
只读 MCP 工具      PASS    get_employee_assets ok=True 耗时=10.2ms
写操作 HITL · 批准  PASS    请假申请已提交，单号 LV-202611-5BC60，共 3 天，当前状态 pending_approval，下一审批人...
写操作 HITL · 拒绝  PASS    已按你的要求取消，本次操作未在内部系统中提交任何变更。
越权调用（结构化错误）    PASS    code=permission_denied
跨租户隔离          PASS    知识库中未收录该信息
------------------------------------------------------------------------------------------------
合计 8 项，通过 8 项
```

---

## 二、架构

```
                       ┌──────────────────────────────┐
   浏览器 ────────────▶ │  Web 工作台 (Streamlit)      │
                       │  流式对话 / 工具卡片 / 引用   │
                       │  HITL 审批弹窗               │
                       └───────────┬──────────────────┘
                                   │ SSE
   ┌───────────────────────────────▼───────────────────────────────┐
   │  Runtime Service（执行面，8001）                                │
   │                                                               │
   │  FastAPI ── 鉴权 / 结构化日志 / /health /ready /metrics        │
   │     │                                                         │
   │  AgentRunner ── 事件翻译（node/token/tool_call/interrupt…）    │
   │     │                                                         │
   │  LangGraph 状态图 ◀── PostgreSQL/SQLite checkpointer（可恢复） │
   │     route ─┬─ knowledge → retrieve → answer                   │
   │            └─ operation → agent ⇄ tools（max_iterations=8）    │
   │                          └→ approval（interrupt）             │
   │            fallback（连续失败/超预算）                         │
   │     │                        │                                │
   │  Hybrid RAG              MCP Client ── stdio/http ──▶ MCP Server│
   │  Dense+BM25→RRF→Rerank   (Schema/scope/熔断/超时/重试)          │
   └───────┬───────────────────────────────┬───────────────────────┘
           │ 模型 Catalog / 审计            │ 内部系统（资产/工单/OA）
   ┌───────▼───────────────────────────────▼───────────────────────┐
   │  Platform API（控制面，8000）                                  │
   │  JWT 签发 / 租户与项目隔离 / 模型 Catalog / 审计日志 / 用户管理  │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
                         PostgreSQL  Redis
```

详细设计（事件协议、数据模型、权限矩阵、图拓扑、扩展点）见 **[docs/architecture.md](docs/architecture.md)**。

---

## 三、快速启动

### 方式 A：Docker Compose（全栈）

```bash
cp .env.example .env          # 至少改 PLATFORM_JWT_SECRET
docker compose up -d --build  # postgres + redis + platform + mcp + runtime + web
docker compose ps             # 等 healthcheck 全绿
open http://localhost:8501    # 工作台
```

### 方式 B：本机跑（不需要 Docker）

```bash
pip install -r requirements.txt

# 1) 控制面（默认 SQLite，首次启动自动建表 + 灌演示账号/模型 Catalog）
python -m uvicorn platform_api.main:app --port 8000

# 2) 执行面（默认 SQLite checkpointer；MCP 走 stdio 子进程）
MCP_OPS_COMMAND=python MCP_OPS_ARGS="-m,runtime.mcp_layer.server,--transport,stdio" \
  python -m uvicorn runtime.main:app --port 8001

# 3) 工作台
python -m streamlit run web/app.py
```

**没有 API Key 也能跑全链路**：`RUNTIME_LLM_API_KEY` 为空时自动切到规则引擎（`OfflineRuleChat`），
路由、工具、HITL、引用、预算、日志全部照常，只是"生成"这一步变成抽取式作答。
配上 Key（任意 OpenAI 兼容端点，默认 DeepSeek）即切换为真实模型。

### 演示账号

| 账号 | 密码 | 角色 | 能看到什么 |
|---|---|---|---|
| `zhangwei@corp.local` | `demo12345` | 普通员工 E1024 | 公开制度文档；只能查自己的资产 |
| `lijing@corp.local` | `demo12345` | HRBP E2048 | 额外可见《薪酬与福利管理细则》（confidential） |
| `admin@corp.local` | `Copilot@2026` | 平台管理员 | 全权限；**不能**代他人请假（需 HR） |

同一句"补充医疗保险报销上限是多少"，员工拿到的是《差旅与报销》，HR 拿到的是《薪酬与福利》——
这是配套演示里最直观的一条。

---

## 四、实测数字

### 4.1 检索质量（50 条评估集，`python -m eval.run_eval`）

语料 10 篇 / 43 chunks。评估集分两半：25 条关键词型 + 25 条改写型（与原文几乎无字面重合，
专门检验语义召回，不然纯词面方法也能刷满分）。CI 是 2000 次 Bootstrap（固定 seed）。

| 管线 | Recall@5 | 95% CI | MRR | nDCG@5 | 关键词子集 | 改写子集 | p50 |
|---|---|---|---|---|---|---|---|
| Dense | 0.960 | [0.900, 1.000] | 0.911 | 0.923 | 1.000 | 0.800 | 1.98 ms |
| BM25 | 0.940 | [0.860, 1.000] | 0.897 | 0.908 | 1.000 | 0.700 | 0.04 ms |
| RRF | 0.960 | [0.900, 1.000] | 0.915 | 0.926 | 1.000 | 0.800 | 1.96 ms |
| RRF + Rerank | 0.960 | [0.900, 1.000] | 0.902 | 0.916 | 1.000 | 0.800 | 2.33 ms |
| **RRF + Rerank + MQ** | **1.000** | [1.000, 1.000] | 0.907 | **0.930** | 1.000 | **1.000** | 2.39 ms |

怎么读这张表（不粉饰）：

- **多查询的收益全部来自改写子集**：0.800 → 1.000，代价是 p50 +0.06 ms（只扩展 BM25，不动
  embedding）。它补的是"员工口语 vs 制度术语"的鸿沟——"内网"→"VPN/远程接入"、
  "涨工资"→"调薪/普调"。MRR 略降 0.007 是因为多召回了两条边缘片段，Recall 的提升更值钱。
- **但这 1.000 有评估集过拟合成分**：同义词表是看到两条未命中样本之后补的。正确做法是留出集
  复测，本文不把它包装成泛化提升。生产上应该从检索日志里挖"零命中查询"来维护词表，而不是继续手补词。
- **Rerank 让 MRR 掉了 0.013**：它牺牲一点排序换精确率——未命中从"混进 2 条噪声"变成
  "明确不答"。在"公司有没有太空电梯补贴"这类问题上，没有 Rerank 会拿《数据分级规范》硬答，
  有了它直接说"知识库未收录"。这是**取 Precision 舍一点 MRR** 的主动选择。
- **绝对分数被语料规模限制了**：43 个 chunk 里取 top5，Dense 也能到 0.96。真实语料（万级）
  上这套管线才有意义的对比，见"局限"。

### 4.2 延迟（本机，离线规则引擎，非真实 LLM）

| 环节 | p50 | 说明 |
|---|---|---|
| 检索（dense 1.97 + sparse 0.07 + fusion 0.02 + rerank 0.37 + 改写 0.01） | **2.44 ms** | `/v1/rag/retrieve` 的 trace 字段可逐段核对 |
| 单次 MCP 工具调用 | **7–10 ms** | 复用长连接后；早先每次 1.5 s（见"踩到的坑 3"） |
| 知识问答端到端 | 15 ms | 含检索 + 作答 + SSE 序列化 |
| 工具路径端到端 | 17 ms | 含 MCP 往返 |

真实 LLM 下这两行会变成 1–3 s，瓶颈全部在模型侧；这里给的是**编排层自身开销**，
用来证明"平台没给你加戏"。

### 4.3 工程指标

- `pytest`：**131 passed**，零 API Key、零外部服务（mock embedding + 脚本化 LLM）。
- `ruff check .`：**0 error**（E/F/I/UP/B；FastAPI 的 `Depends` 默认参数按框架约定豁免 B008）。
- `scripts/eval_gate.py`：Recall@5 ≥ 0.95、MRR ≥ 0.85 才允许合并，挡住"改检索反而更差"。
- 覆盖：RAG（权限过滤/融合/重排/阈值/查询改写）、MCP（Schema 不泄露 caller、越权、幂等、熔断、
  写操作不重试、工具契约快照）、图（知识/工具/批准/拒绝/兜底/预算）、控制面（登录、租户隔离、
  审计、401/403）、guardrails（注入/密钥/PII/跨租户金丝雀）、UI 流协议、trace 透传。
- 预算：单次运行 `$0.05` 上限，超限 HTTP 402 并降级到 cheap 模型；每次调用都记账（tokens + USD）。
- 追踪：一条请求的 `trace_id` 贯穿 HTTP → 图 → LLM → 工具，span 属性用 OTel GenAI semconv 命名。

---

## 五、模块实现对照

### A. 混合检索 RAG（`runtime/rag/`）

- **解析层**：PDF（pypdf）/ Markdown（带 front-matter）/ HTML（BeautifulSoup），统一清洗成 `Document`。
- **切分**：`SemanticChunker`（按句向量相似度断点 + 百分位阈值）和 `RecursiveChunker`（分隔符递归），
  `RAG_CHUNKER=semantic|recursive` 一行配置切换，评估脚本 `--chunker` 可对比。
- **召回**：Dense（可插拔 embedder：hashing / BGE / OpenAI）+ BM25-Okapi（中文 bigram），并行跑。
- **融合**：标准 RRF `score += 1/(k + rank)`，`k=60`。
- **重排**：`CrossEncoderReranker`（BGE-Reranker）/ `HeuristicReranker`（零模型，默认）。
  启发式这层用的是 **IDF 主导的证据强度**而非简单词覆盖，理由见 `rerank.py` 注释与"踩到的坑 2"。
- **权限**：`AccessContext(tenant_id, user_id, roles, scopes)` → 作为 predicate 传给召回器，
  **在排序之前过滤**（不是取回后再筛，否则 top_k 会被别人的文档挤满）。缺 `tenant_id` 的 chunk 一律不可见（fail-closed）。
- **兜底**：低于 `score_threshold` 全部丢弃，宁可答"未收录"也不拿噪声硬编。
- **增量索引**（`runtime/rag/lifecycle.py`）：**双指纹**判定变更——`content_hash` 只看正文（变了才重嵌），
  `policy_hash` 只看 `tenant_id/classification/allowed_scopes/allowed_roles`（变了必重写元数据）。
  两者分开的理由见"踩到的坑 8"：只按正文哈希，改密级会被判成"没变"而跳过。
  改 1 篇只重嵌 1 篇（实测 44 chunk → 6 chunk）；源文件删除时 `--prune` 连 chunk 一起清，
  且向量库与 BM25 同时删。清单与库不一致（重启没落盘）时自动退化为全量重建，见"踩到的坑 9"。
- **查询改写**（`RAG_QUERY_TRANSFORM=multi_query`）：领域同义词优先（口语→制度术语），
  扩展词只来自受控词表（不用伪相关反馈，原因见"踩到的坑 7"）；只扩展 BM25，不增加
  embedding 调用。重排只在词表命中时使用扩展词，避免查询漂移。
- **评估**：`eval/metrics.py` 提供 Recall/MRR/nDCG/MAP/P@5 与 Bootstrap CI；五路对比见 4.1。

### B. MCP 标准化工具层（`runtime/mcp_layer/`）

- `InternalOpsServer`（FastMCP）暴露 5 个工具：`get_employee_assets`、`query_ticket_status`、
  `search_directory`、`submit_leave_request`、`create_ticket`。
- 每个工具入参是 Pydantic 模型，出参统一 `OpsResult{ok, data, error{code, message, details}}`；
  **业务失败也是 200 + 结构化 error**，不抛裸异常、不把内部堆栈漏给模型。
- **caller 不进 tool schema**：模型看不到、也改不了"我是谁"，caller 由运行时从 JWT 解出再注入。
- 权限两维：`scope` 决定能不能用这个工具，`role` 决定能看到谁的数据（员工只能查自己的资产，HR/IT 才能查他人）。
- 韧性链 `CircuitBreaker → Timeout → Retry`：只读工具最多重试 2 次（指数退避 + 抖动），
  **写操作绝不自动重试**（超时重试 = 两张请假单），靠 `idempotency_key` 去重。
- 装配：`MultiServerMCPClient` + **长连接会话复用**，工具白名单由 `MCP_ALLOWED_TOOLS` 控制。
- **工具注解**：5 个工具都带 MCP 2025-06-18 的 `readOnlyHint / destructiveHint / idempotentHint /
  openWorldHint`；客户端不用理解业务就能判断"这是不是写操作、会不会删数据"。
- **契约可 diff**：`GET /v1/tools` 输出 inputSchema / outputSchema / 16 位 schema_hash /
  contract_version；`tests/test_tool_contract.py` 做快照比对，改参数必须显式更新快照。

### C. LangGraph 状态图与 HITL（`runtime/graph/`）

```
START → route ─┬─ knowledge → retrieve ─┬→ answer → END
               │                        └→ fallback → END
               └─ operation → agent ─┬─ 写操作 → approval ─┬ approve/edit → tools
                                     │                    └ reject → END（终止）
                                     ├─ 只读 → tools → agent（循环，≤8 轮）
                                     ├─ 无工具调用 → END
                                     └─ 预算/异常 → fallback → END
```

- **条件路由**在入口就分叉，不是"先检索再看要不要用工具"。
- **工具循环有硬上限**：`max_iterations=8`，且 `recursion_limit` 兜底；超限后模型只能基于已有结果收尾。
- **HITL 用 `interrupt()` 实现**（不是 `interrupt_before`），中断负载里带完整参数供人审；
  支持 `approve / edit（改参数后继续）/ reject`。
- **拒绝是终止态**：不回到 agent。否则模型看到"工具报错"很可能换个参数再请求一次，等于绕过刚被拒绝的审批。
- **状态持久化**：PostgreSQL（生产）/ SQLite（本地）checkpointer。已实测"中断后杀掉进程、重启、继续审批"全链路可恢复。
- **兜底**：连续 2 轮工具全失败 → 兜底节点给"系统繁忙已记录"类安全话术，不把异常抛给用户。

### D. 可观测性、成本与测试（`runtime/obs/`）

- **结构化日志**：中间件注入 `request_id`，业务日志自动带 `user_id / tenant_id / thread_id`，全 JSON 一行一条。
- **指标**：`/metrics` 暴露 `agent_run_total`、`agent_node_total`、`tool_call_total`、`agent_run_ms`（含 p50/p90/p99）。
- **追踪**：LangSmith / Langfuse 双通道；同时自实现 GenAI semconv span（`invoke_agent` → `chat`
  → `execute_tool`），`trace_id` 从 W3C `traceparent` 复用，响应头回写 `x-trace-id`，
  span 字段可直接被 collector / Langfuse / Phoenix 消费。
- **预算**：`BudgetGuard` 逐次记账，超限抛 `BudgetExceeded` → HTTP 402 或降级 cheap 模型。
- **健康探针**：`/health`（存活）、`/ready`（checkpointer + MCP + RAG 三项依赖逐项体检），compose 用它们做 `depends_on`。

### E. 安全护栏与前端协议（`runtime/guardrails/`、`runtime/api/ui_stream.py`）

- **输入闸门**（阻塞式，在拉 Catalog / 调模型之前）：覆盖指令、套取系统提示、越狱、绕过审批、
  密钥、超长输入 → 直接拦截并返回固定安全话术，`usage.spent_usd=0`。
  手机号/身份证/银行卡只脱敏不拦截（用户可能只是在陈述事实）。
- **输出闸门**：跨租户金丝雀 `ZQ-OTHER-TENANT`、密钥、系统提示特征串 → 替换为拦截话术，
  `status=blocked`，并推一条 `guardrail` 事件供告警。判定结果只回传规则名，不回传原文。
  流式输出先过 `StreamGuard`：风险前缀出现即扣住后续 token（宁可晚 8 个字符，也不发半截密钥），
  流结束后再对完整 answer 校验一次。
- **工具参数闸门**：模型自带 `caller` 或参数夹带密钥 → 不执行工具，返回 `guardrail_blocked`。
- **事件协议**：内部仍是领域事件（`route/retrieval/tool_call/interrupt`）；新增
  `/v1/chat/ui-stream` 与 `/v1/threads/{id}/resume/ui-stream`，按 Vercel AI SDK
  UI Message Stream v1 输出（`text-start/delta/end`、`tool-input-available`、
  `tool-approval-request`、`data-*`、`[DONE]`），前端可直接用 `useChat`。

### F. 日期与参数理解（`runtime/nlp/dates.py`）

离线模式没有模型，所以"听懂人话"这件事必须由可单测的纯函数承担：

- **抽取**：`2026-11-03` / `2026-11-3` / `2026/11/3` / `2026.11.3` / `2026年11月3日` /
  `11月3日` / `9日` / `明天` 都能识别。只认四种相对日期，不做"下周三"这类需要日历推断的表达——
  推断就有猜错的可能，猜错就是一张错单。
- **补全**：缺年按当年、`11月30日到3日` 按跨月（12-03）、`12月25日到1月5日` 按跨年、
  `从明天起请3天` 由时长推算结束日。**真倒序与日历上不存在的日期不猜**，直接要用户确认。
- **追问**：缺哪个说哪个（"已记下开始日期 2026-11-03，还需要结束日期"），并给三句能照抄的示例，
  而不是丢一个格式要求。
- **多轮**：`route_node` 在"上一轮是操作、本轮只补了日期/时长"时延续意图，
  否则同一次请假会被路由拆成两半（见坑 11）。
- **事由清洗**与解析共用同一套抽取规则，避免审批卡片里出现"因 2026-11-03 到…请假"这种半成品。
- 这套逻辑与是否接真模型无关：路由延续在两种模式下都生效；接真模型后日期抽取由模型完成，
  但"补参数的那一轮该走哪条分支"仍由这里决定。

---

## 六、踩到的坑（都是真实修过的，不是设想）

1. **interrupt 后立刻 resume 会报"没有待审批的操作"**
   —— 流式消费在 `interrupt` 事件处 `break`，把 `astream` 生成器扔在半路，
   LangGraph 还没来得及把 `next=("approval",)` 落盘。50 ms 后就正常，所以人工点按钮永远测不出来，
   自动化脚本一跑就炸。改成把流读完，用终止事件合并审批负载。

2. **相关性阈值调了三次才对**
   —— 第一版"数命中几个查询词"：问"公司有没有太空电梯补贴"，仅凭命中"公司"就拿到 0.17，
   越过阈值把《数据分级规范》送进上下文。第二版"IDF 加权覆盖率"：又被问句里的冗余
   bigram（"准是""是多"）稀释，真问题"出差住宿标准是多少"反而掉到阈值下。
   最终版拆成 **证据强度（命中词的最高 IDF）+ 覆盖率** 两层，噪声 0.49 / 正常问题 ≥0.60，阈值 0.55。

3. **每次 MCP 调用 1.5 秒**
   —— `langchain-mcp-adapters` 的 `tool.ainvoke()` 每次都新建 ClientSession，stdio 传输下等于
   **每次调用重新拉起一个 Python 进程**。改成自己持一条长连接会话并做断线重建，降到 7–10 ms（约 200×）。

4. **只读工具 scope 通过 ≠ 能查任何人**
   —— `get_employee_assets(E2048)` 用员工身份能查到 HR 的资产：工具级 scope 校验通过了，
   但 `employee_id` 是模型填的，模型完全可以把"我的资产"解析成别人的工号。补了"本人或 HR/IT"的角色判定。

5. **拒绝后模型会再试一次**
   —— 拒绝曾经走回 agent，模型看到 `rejected_by_user` 报错，换个参数重新请求审批，
   等于把被拒的操作又提交了一遍。改为拒绝即终止，系统给固定话术。

6. **演示账号权限改了但库里没变**
   —— 给 `hr` 角色补 `chat:write` 后，老库里的账号还是旧 scope，登录即 403。
   bootstrap 现在对演示账号做作用域对账（只动演示账号，不动真实用户）。

7. **伪相关反馈把"太空电梯"答成了年假**
   —— 多查询第一版用 RM3：取首轮 BM25 top-3 文档的高 IDF 词扩展查询。问"公司有没有太空电梯补贴"，
   BM25 靠"公司/有没有"误召回年假制度，扩展词全部来自那篇文档，重排覆盖率被刷到 1.0，
   噪声成了"满分证据"，最后答出"年假 10 天"。改为**受控词表扩展**：词表没命中就只查原问题；
   重排也只在词表命中时使用扩展词。这是"检索链路会自我强化"的典型案例，调阈值救不回来。

8. **增量索引只按正文哈希，把机密文档降级成了公开**
   —— 做增量时最自然的写法是 `sha256(doc.text)` 比一比，没变就跳过。
   于是把一份制度从 `internal` 上调为 `confidential` 后，正文一字未改 → 判 unchanged → 整体跳过 →
   库里留着旧的低密级元数据 → **普通员工照样检索得到**。整个过程不报错、不告警、日志一片绿，
   只会在某次合规检查时被发现。
   修法是把指纹拆成两个：`content_hash`（正文，决定要不要重嵌）和 `policy_hash`
   （`tenant_id`/`classification`/`allowed_scopes`/`allowed_roles`，决定要不要重写元数据）。
   `updated_policy` 成为独立的变更类型并单独计数——"权限口径变了"在审计上是安全事件，不该混进内容更新。

9. **清单说"都没变"，库却是空的**
   —— 增量状态存在 `index_manifest.json`，向量存 `vector_store.json`。内存后端重启、忘了 `--persist`、
   换台机器，都会出现"清单还在、库空了"。此时若相信清单，同步会全部跳过，**得到一个空索引却报成功**，
   线上表现就是"知识库突然什么都查不到"。现在同步前先比对库里的 chunk 数与清单，不一致就退化为全量重建
   并打 `index_sync_forced` 警告；清单 JSON 损坏也当不存在处理。
   连带一个反向坑：`all unchanged` 时退出码不能是 1，否则 CI 会把"什么都没变"当成失败。

10. **"2026年11.3 到 2026.11.9" 被回了一句"格式 YYYY-MM-DD"**
   —— 用户连试三种写法都被同一句话挡回来。根因不在模型，而是规则引擎里只有一条
   `re.compile(r"\d{4}-\d{2}-\d{2}")`：点分隔不认、中文"年/月"不认、单位数月日也不认。
   更糟的是这句追问是硬编码的，用户不知道还能怎么说。
   修法是把日期理解拆成三层放进 `runtime/nlp/dates.py`：**抽取**（`2026-11-3`/`2026.11.3`/
   `2026年11月3日`/`11月3日`/`明天` 一视同仁）→ **补全**（缺年补年、`11月30日到3日` 跨月、
   `12月25日到1月5日` 跨年、`从明天起请3天` 推算结束日）→ **追问**（缺哪个就说哪个，
   并给三句能照抄的例子）。真倒序（`2026-11-09 到 2026-11-03`）和日历上不存在的日期
   仍然报错——猜一个等于替用户签单。

11. **多轮补全不只是解析问题，路由也会把"补参数"当成新问题**
   —— 上一轮问"帮我提交年假从 2026-11-3"，系统追问结束日期；用户回"到11月9日"，
   结果去查了一通年假制度。因为 `classify_intent` 只看最后一条消息，"到11月9日"既没有
   动作词也没有疑问词，被判成知识问答，同一次请假被拆成两半。
   修法是让补参数的一轮**延续上一轮的意图**：上轮是 operation 且本轮"只给了日期/时长、
   没提新问题"时，继续走 operation。边界也守住了——上轮在问制度、本轮说日期，仍然走知识问答。
   注意这条与是否接真模型无关：路由是规则层，两种模式都会踩。



---

## 七、目录

```
common/            跨服务契约：scopes 常量、JWT、错误信封、JSON 日志、trace 上下文中间件
platform_api/      控制面：auth / projects / models catalog / users / audit / health
runtime/
  api/             SSE 对话、UI Message Stream、HITL resume、工具目录、检索调试、健康与指标
  graph/           state / nodes / builder / runner / checkpointer / registry
  rag/             parsers / chunkers / embeddings / bm25 / fusion / query_transform / rerank / access / pipeline
  mcp_layer/       server(FastMCP) / client / catalog / schemas / policies / resilience / backends
  guardrails/      输入 / 输出 / 工具参数三层拦截（tripwire 语义）
  llm/             client(OpenAI 兼容 + 离线规则) / gateway(预算+降级) / catalog
  obs/             budget / metrics / tracing / spans(GenAI semconv)
web/app.py         Streamlit 工作台（流式、工具卡片、引用、审批弹窗）
eval/              50 条评估集 + 五路管线对比 + nDCG/MAP/P@5/CI
scripts/smoke.py   端到端 8 场景冒烟；eval_gate.py 评估门禁
tests/             131 个单测（零 API Key，含工具契约快照）
docs/architecture.md
docs/github-benchmark.md   14 个 GitHub 项目的逐条对标与差距清单
```

---

## 八、局限（写清楚比藏起来好）

- **语料规模小**：43 chunks 的评估集区分度有限，Dense 就能到 0.96。真实上量后
  混合检索/重排的差距才会明显；那时阈值必须重新标定（换 CrossEncoder 后分数量纲完全变了）。
- **同义词表有过拟合风险**：`RAG_QUERY_TRANSFORM=multi_query` 的 1.000 是在 50 条评估集上
  得到的，其中两条未命中样本促成了词表补充。真实项目应保留 held-out 集，或改用检索日志挖词。
- **Guardrails 是规则不是语义模型**：能挡住已知的注入模板与密钥格式，挡不住"换一种说法"的
  语义绕过。接真实模型后应在这一层加一个便宜的 judge，接口已留好（`GuardrailEngine`）。
  另外，流式下的 PII 脱敏只作用于最终 `answer`，不承诺逐 token 脱敏（密钥/金丝雀是逐 token 拦的）。
- **追踪是"格式对齐"而非"协议接入"**：span 字段按 OTel GenAI semconv 命名，但没有内置
  OTLP exporter（避免拖入整套 SDK）。接 Langfuse/Phoenix 需要 collector 或 sidecar 转换。
- **离线规则引擎不是 LLM**：没 Key 时"生成"是抽取式的，能跑通链路、能验证权限与编排，
  但答不了需要归纳/推理的问题。它的存在是为了让 CI 和演示不依赖外部服务。
- **Embedding 默认 hashing**：零依赖、可复现，但语义能力远不如 BGE。`RAG_EMBEDDING_PROVIDER=sentence-transformers`
  可切真模型（需额外安装，索引需重建）。
- **Redis 目前只做预留**：会话状态在 checkpointer 里，Redis 用于后续的限流/分布式锁/事件总线。
- **前端是演示级**：Streamlit 单页，够展示流式/工具卡片/审批，不适合做多租户后台。
- **未做**：多 Agent 子图编排、工具结果的长期记忆、增量索引的向量复用（正文未变但密级变更时会重嵌一次）、A/B 灰度、真正的向量库压测。

---

## 九、常用命令

```bash
make test      # 131 个单测（零 API Key）
make lint      # ruff，0 error
make eval      # 五路管线对比 + Recall/MRR/nDCG/MAP/P@5 + 95% CI
make gate      # 评估门禁：Recall@5 ≥ 0.95 且 MRR ≥ 0.85
make up        # 全栈 compose
python scripts/smoke.py          # 8 场景端到端
python scripts/verify_dates.py   # 5 场景：请假日期的多种写法与多轮补全

# 灌库（默认增量：没改的文档不重嵌）
python -m runtime.rag.ingest --tenant demo --path data/knowledge --persist
python -m runtime.rag.ingest --tenant demo --path data/knowledge --dry-run   # 先看会动什么
python -m runtime.rag.ingest --tenant demo --path data/knowledge --prune     # 源文件删了就清索引
```
