# 架构设计

本文只写"为什么这样切"，不重复 README 的功能清单。目标是让接手的人能在半小时内
判断:哪块该改、改了会牵动谁。

---

## 1. 为什么是两个进程

```
platform_api（控制面）           runtime（执行面）
─────────────────────           ─────────────────
谁可以进来                       进来之后怎么干活
JWT / 租户 / 项目 / 模型目录      图执行 / 工具 / 检索 / 流式
低频、强一致、可审计               高频、长事务、可重试
改一次部署一次                    改一次影响所有租户
```

三条边界规则，仓库里靠目录和依赖方向保证:

1. **控制面不含任何 LangGraph 逻辑**。`platform_api/` 里没有 `langgraph` import。
2. **执行面不直接读写控制面的库**。要模型目录、要审计，都走 HTTP(`runtime/llm/catalog.py`、`platform_api/routers/audit.py`)。
   这样执行面可以横向扩 N 个副本而不会和控制面抢连接池。
3. **跨服务的常量只放 `common/`**。`scopes.py` 里的 `chat:write` / `ops:leave:write` 是控制面签发、MCP Server 校验的同一份契约。
   历史上这类字符串最容易出现"MCP 认识的 scope 平台不认识"，所以只允许有一处定义。

代价是:一次对话要多一跳(执行面 → 控制面拿模型目录)。为此加了 60 s 正缓存 + 30 s 负缓存,
控制面挂了也不会每次对话都等超时。

---

## 2. 数据模型(控制面)

```
Tenant 1──N User
   │
   ├──N Project(graph_key, collection, config)
   ├──N ModelCatalogEntry(model_key, model_name, price_in/out, is_default, is_cheap)
   └──N AuditLog(action, resource, outcome, request_id)
```

`User` 上有两个身份维度，别混:

- `roles`: `admin / hr / employee / auditor / viewer`,决定**业务角色**(能看到谁的数据);
- `scopes`: 由角色推导的权限串,决定**能不能调这个接口/工具**(`chat:write`、`ops:leave:write`、`rag:confidential`)。

`employee_no`(工号)是平台账号 ↔ 内部系统的映射点:JWT 里带 `emp` claim,执行面把它塞进 `CallerContext.user_id`,
MCP Server 用它做"只能查本人"的判定。

会话状态不在这个库里。它在 LangGraph 的 checkpointer 表里(见 §5),
因为那是框架的 schema,混进业务库只会让迁移变成噩梦。

---

## 3. 事件协议(对外唯一接口形态)

前端只认 `event:` 字段,不解析内部对象。`/v1/chat/stream`(SSE) 与 `/v1/chat`(JSON) 的事件序列完全一致,
后者只是把 SSE 收成数组,方便 curl / 单测 / 自动化。

| event | 何时发 | data 关键字段 |
|---|---|---|
| `start` | 运行开始 | `thread_id, mode(new/resume), offline_llm, model_source, trace_id` |
| `route` | 意图判定完 | `intent: knowledge \| operation` |
| `retrieval` | 检索完 | `chunks, filtered_out, timings_ms{dense,sparse,query_transform,fusion,rerank}` |
| `citations` | 有引用时 | `[{title, snippet, score, section, source}]` |
| `token` | 增量文本 | `delta` |
| `tool_call` | 即将调用 | `calls:[{tool, args, requires_approval}]` |
| `tool_result` | 工具返回 | `{tool, ok, data\|error, latency_ms}` |
| `interrupt` | 等人工审批 | `thread_id, approval:{calls:[...]}` |
| `approval_result` | 审批结果 | `action: approve \| edit \| reject` |
| `guardrail` | 安全闸门触发/脱敏 | `stage(input\|output\|tool), action, rules, redactions` |
| `fallback` | 进兜底 | `reason` |
| `node` | 节点推进 | `{node, status}` |
| `done` | 正常结束 | `answer, citations, usage, awaiting_approval, iterations, latency_ms, trace_id, guardrail` |
| `error` | 异常 | `code, message, status` |

三条语义约定,前端必须依赖:

1. **终止态唯一**。一轮对话只会有一个终止事件:`done` / `interrupt` / `error`。
   `interrupt` 之后不会再有 `done`(实现上把审批负载合并进了终止事件,见 `runtime/api/chat.py`)。
2. **`interrupt` 不等于失败**。它表示"图停在这里等人",`awaiting_approval=true`,可 `POST /v1/threads/{id}/resume`。
3. **`token` 可能为空**。知识路径在没有检索命中时走的是"未收录"话术,不调模型。
4. **`guardrail` 不是终止态**。输入被拦时后面一定跟 `done(status=blocked)`;输出被拦时它出现在
   `done` 之前,`done.answer` 已被替换为安全话术,不会把原文返回给前端。

### 3.1 前端兼容层

`/v1/chat/ui-stream` 与 `/v1/threads/{id}/resume/ui-stream` 把同一份事件翻译成
Vercel AI SDK UI Message Stream v1:文本走 `text-start/delta/end`,工具走
`tool-input-available/output-available`,写操作额外发 `tool-approval-request`,
自定义信息走 `data-citations/data-guardrail/data-approval`,结束 `data: [DONE]`。
两条路径共享同一个 `runner.stream()`,不存在"两套编排逻辑"。

---

## 4. 图拓扑与状态

```
START
  │
route ─────────────── 判 intent(纯问答 / 业务操作)
  │
  ├─[knowledge]→ knowledge_retrieve ─┬→ knowledge_answer → END
  │                                  └→ fallback → END
  │
  └─[operation]→ agent ─┬─[有待审批写操作]→ approval ─┬ approve/edit → tools
                        │                            └ reject → END
                        ├─[只有只读]→ tools ─→ agent(循环,≤8 轮)
                        ├─[无工具调用]→ END
                        └─[预算/异常]→ fallback → END
```

状态字段(`runtime/graph/state.py`)分四组,只有**可序列化**的东西能进 state:

```
身份   tenant_id user_id thread_id roles scopes
路由   intent citations retrieved_block
循环   pending_tool_calls tool_events iterations consecutive_failures approvals
结果   answer status fallback_reason degraded usage
```

运行时能力(`LLMGateway`、`Toolbox`、`CallerContext`)**不进 state**,
而是通过 `config["configurable"]["deps"]` 注入。原因很实际:第一次上 PostgresSaver 时,
往 state 里塞一个工具对象会直接序列化失败;就算塞得进去,checkpoint 也会变成一堆无法迁移的 pickled 对象。

### 四条防止"图失控"的护栏

| 护栏 | 值 | 触发后行为 |
|---|---|---|
| `max_iterations` | 8 | 超限后不再执行工具,把已有结果交给模型收尾 |
| `recursion_limit` | `max_iterations*6+20` | 框架级硬停,防条件边写错导致死循环 |
| `max_consecutive_failures` | 2 | 连续两轮工具全失败 → 兜底话术 |
| `run_budget_usd` | 0.05 | 超限 `BudgetExceeded` → 402 或降级 cheap 模型 |

### 4.1 路由为什么要看上一轮

`route` 只看最后一条用户消息。这在单轮里没问题,但"补参数"的一轮会把同一次业务拆成两半:

```
用户: 帮我提交年假从 2026-11-3     → route=operation → agent → 缺结束日期,追问
用户: 到11月9日                    → route=knowledge → 去查年假制度(错)
```

"到11月9日"既没有动作词也没有疑问词,按单条规则必然落到 knowledge。
所以 `classify_intent(text, previous=...)` 加了一条**受限的意图延续**:

> 上一轮是 operation,且本轮"只给了日期/时长、没提新问题" → 继续 operation。

三个边界都守住了:
- 上一轮在问制度,这一轮说日期 → 仍是 knowledge(不能因为说过"年假"就把提问变成操作);
- 本轮自带疑问词("年假有几天") → 仍走 knowledge,哪怕上一轮是操作;
- 没有上一轮 operation → 不延续。

这条与是否接真模型无关:路由是规则层,两种模式都会踩;接真模型后日期抽取交给模型,
但"这一轮该进哪条分支"仍由这里决定。`runtime/nlp/dates.py` 负责把各种写法解析成
`(start, end)`,两者的分工是"进哪条路"与"参数是什么",不重叠。


---

## 5. 持久化与"可恢复"的真实含义

checkpointer 三档,由 `RUNTIME_CHECKPOINT_BACKEND` 选:

| 值 | 用途 | 行为 |
|---|---|---|
| `postgres` | 生产 | `AsyncPostgresSaver`,启动时 `setup()` 幂等建表 |
| `sqlite` | 本地/演示(默认) | `AsyncSqliteSaver`,`./var/runtime.db` |
| `memory` | 单测 | 进程退出即丢,`APP_ENV=prod` 时启动即报错 |

`is_production_safe()` 会在 `APP_ENV=prod` 且不是 postgres 时打 error 日志 —— 不是警告,是 error。
"禁止用内存 runtime 上生产"这条约束如果只写在 README 里,三个月后一定会被违反。

**已验证的恢复路径**:触发 HITL 中断 → `kill` 执行面进程 → 重启 → `GET /v1/threads/{id}` 仍是
`awaiting_approval` 且带着原参数 → `POST /resume` 批准 → 工具正常执行。
这件事之所以成立,是因为审批所需的一切(pending_tool_calls、messages、approvals)都在 checkpoint 里,
不在进程内存中。

---

## 6. MCP 工具层

### 6.1 一次工具调用的完整链路

```
Agent 决定调用 submit_leave_request
        │
   deps.toolbox.call(name, args, caller)          ← caller 来自 JWT,模型不可见
        │
   ① 本地 scope 预检(省一次往返,错误更早)
   ② 注入 caller + idempotency_key(写操作)
   ③ CircuitBreaker.before_call()   熔断中 → 直接 circuit_open
   ④ asyncio.wait_for(..., timeout) 只读 8s
   ⑤ 失败重试(只读 ≤2 次,指数退避+抖动;写操作 attempts=1)
        │
   MCP Client(长连接会话) ──stdio/http──▶ MCP Server
        │
   Server: Pydantic 校验 → _guard(scope) → 角色判定(本人/HR) → 业务后端
        │
   返回 OpsResult{ok, data|null, error{code,message,details}}
        │
   统一归一成 ToolCallOutcome → ToolMessage + tool_result 事件
```

### 6.2 为什么错误要"结构化返回"而不是抛异常

三类失败必须能被上层区分对待:

| error.code | 语义 | 上层行为 |
|---|---|---|
| `permission_denied` | 权限不足 | 直接告诉用户缺什么,不重试 |
| `timeout` / `circuit_open` | 依赖不可用 | 计入连续失败,触发兜底 |
| `validation_error` | 参数不合法 | 把校验信息回给模型让它改参数 |
| `employee_not_found` / `ticket_not_found` | 业务不存在 | 如实回答,不算失败 |

如果一律抛异常,`tools_node` 只能数"失败次数",无法决定"该重试还是该告诉用户"。

### 6.3 会话复用(性能关键)

`langchain-mcp-adapters` 的 `tool.ainvoke()` 每次新建 `ClientSession`;stdio 传输下即**每次调用重启一个 Server 进程**,
实测 1.4–1.5 s/次,全部耗在 Python 解释器启动。`McpToolbox` 改为持有一条长连接会话并在
管道断开时重建,降到 7–10 ms。

---

## 7. RAG 管线

```
文档 ──parsers──▶ 纯文本 ──chunker──▶ chunks ─┬─▶ embedder ─▶ vector store(Dense top-20)
                                              └─▶ BM25 索引  ─▶ BM25(sparse top-20)
                                                        │
                                    权限 predicate(租户/密级/角色/scope) 在**排序前**生效
                                                        │
                                            RRF 融合 k=60 ─▶ Rerank top-5
                                                        │
                                           score_threshold 过滤 ─▶ 上下文块 + citations
```

### 7.1 权限为什么必须"前置"

先取 top_k 再过滤,在"该租户只能看 3 篇"时,top_k=20 会被别人的文档占满,正确结果根本进不了候选。
所以 `chunk_visible()` 作为 predicate 传给召回器,**在打分阶段就排除**。
缺失 `tenant_id` 的 chunk 直接不可见(fail-closed),而不是"没标就放行"。

判定顺序:租户 → 密级(pub/internal/confidential 用"可见等级上限"表达) → `allowed_scopes` → `allowed_roles`。

### 7.2 重排器打分

启发式的分数 = **证据强度 × (0.65 + 0.35 × 覆盖率) × 长度惩罚**:

- 证据强度 = 命中词中的最高 IDF ÷ 候选集理论最大 IDF → 回答"有没有命中一个有辨识度的词";
- 覆盖率 = 命中词 IDF 之和 ÷ 语料内查询词 IDF 之和 → 防止只命中一个稀有词就拿满分。

纯覆盖率方案会被问句里的冗余 bigram("准是""是多")稀释,纯计数方案会让"公司"这种高频词混过阈值。
两者都不是调参能解决的,是打分函数形状不对。

`CrossEncoderReranker` 的分数量纲完全不同(0–1 归一化),切换后 **`RAG_SCORE_THRESHOLD` 必须重新标定**,
这是最容易漏的一步。

### 7.3 阈值与"不答"

`score_threshold=0.55` 是"宁可不答"的开关。低于阈值全部丢弃 → `knowledge_answer` 直接给
"知识库未收录",**不调模型**。这一条同时省了钱、堵了幻觉,代价是少量本该答的没答。
评估集里 2 条改写型 query 就是这么丢的,记在 README 里。

### 7.4 索引生命周期(增量而不是全量重建)

万级语料下,改一份文件就重嵌整个库是不可接受的。`sync_documents()` 用**双指纹**决定做什么:

| 指纹 | 覆盖 | 变了要做什么 |
|---|---|---|
| `content_hash` | 仅正文 | 重新切分 + 重新嵌入 |
| `policy_hash` | `tenant_id` / `classification` / `allowed_scopes` / `allowed_roles` | 向量可复用,但**元数据必须重写** |

两个指纹必须分开,因为它们的失效方式完全不同:

- 若只按正文哈希,把文档从 `internal` 上调为 `confidential` 时正文没变 → 判 unchanged → 跳过 →
  库里留着旧密级 → **普通员工照样检索得到**。这是不报错、不告警的权限降级。
- 若把 `title`/`source` 也算进指纹,改个标题就触发全量重嵌,纯浪费。

所以 `updated_policy` 是独立的变更类型,报告里单独计数(审计上"权限口径变了"是安全事件,不该混进内容更新)。

**状态丢失的防御.** 清单说"都没变"、向量库却是空的(内存后端重启、忘了 `--persist`、换了机器),
若相信清单就会全部跳过,得到一个空索引却报"同步成功"。所以同步前先比对库里的 chunk 数,
不一致就退化成全量重建,并打 `index_sync_forced` 警告。同理:清单损坏 → 当作不存在 → 全量重建。

**下架比少收录更危险.** 源文件删了但 chunk 留着,模型会拿废止条款当依据。`prune=True` 时
连 chunk 一起删,且**向量库与 BM25 必须同时删**——只删一边,另一边照样召回旧内容。

**已知取舍.** 权限变更会重新嵌入同一段正文(正文没变,向量必然一样)。要做向量复用得先让
`VectorStore` 协议暴露 `get_vectors(ids)`,当前场景下不划算,记录在 `docs/github-benchmark.md` 的未做清单里。


---

## 8. 可观测性

```
请求 ──RequestContextMiddleware──▶ request_id 生成/透传 ──▶ contextvar
                                        │
                    ┌───────────────────┼────────────────────┐
                 JSON 日志          Prometheus 指标        追踪
        (service/request_id/      (run_total/node_total   (LangSmith /
         user_id/tenant_id/        /tool_call_total/       Langfuse,
         thread_id/latency_ms)     agent_run_ms 分位)      run_config 注入)
```

- 日志一行一条 JSON,`grep '"level": "ERROR"'` 就能捞异常;`request_id` 能从网关一路串到 MCP Server。
- 指标自己实现(不引 `prometheus_client`),`/metrics` 输出 текст 格式,够 Grafana 抓。
- 追踪挂在 `run_config()` 里,零配置时降级为 no-op 并打一条 `tracing_disabled`,不阻塞业务。

---

## 9. 失败模式与兜底策略

| 失败点 | 表现 | 兜底 |
|---|---|---|
| 控制面不可达 | 拿不到模型目录 | 60 s 正缓存 / 30 s 负缓存 → 回落环境变量配置 |
| MCP Server 起不来 | 工具发现失败 | 启动期降级 `LocalToolbox` 并标 `degraded=true`(运行期不静默降级) |
| 工具持续失败 | 连续 2 轮全失败 | 兜底节点,给"系统繁忙已记录"话术 |
| 模型超预算 | `BudgetExceeded` | HTTP 402 或切 cheap 模型重试 |
| 检索无命中 | `chunks=[]` | 明确"未收录",不调模型 |
| checkpointer 不可用 | 无法恢复会话 | 启动即失败(宁可起不来,也不要起来后 HITL 全崩) |
| 前端断开 SSE | 流被取消 | 图已跑完的部分留在 checkpoint,重新 GET 线程状态即可 |

**一条贯穿原则:运行期不静默降级。** 启动期连不上 MCP 可以降级成本地实现(并显式标记),
但运行期失败一律走熔断 + 兜底,不偷偷换实现 —— 那会让 MCP 的权限边界形同虚设。

---

## 10. 扩展点

| 想改什么 | 改哪里 | 注意 |
|---|---|---|
| 换向量库 | `runtime/rag/vectorstore.py` 实现 `VectorStore` 协议 | predicate 语义要保持"先过滤再排序" |
| 换 embedder | `runtime/rag/embeddings.py` 加 provider + 配置 | 换维度必须重建索引 |
| 换重排器 | `runtime/rag/rerank.py` + `RAG_RERANKER` | 分数量纲变了要重标阈值 |
| 加工具 | `mcp_layer/schemas.py` + `server.py` + `policies.py` | 三处缺一不可:Schema、实现、策略(scope/write) |
| 加图分支 | `graph/builder.py` + `nodes.py` | state 字段要可序列化;新条件边记得给 `END` 留映射 |
| 换前端 | 消费 §3 的事件协议 | 别依赖 `token` 一定非空 |
| 多租户隔离升级 | `runtime/rag/access.py` + `common/scopes.py` | 别在 router 里散写权限判断 |
| 接真实 OA/ITSM | `mcp_layer/backends.py` | 保持 `OpsResult` 契约,别把 HTTP 异常往上抛 |
| 加安全规则 | `runtime/guardrails/core.py` | 只回传规则名,不回传命中原文;输出闸门不替代权限过滤 |
| 扩查询改写 | `runtime/rag/query_transform.py` | 扩展词只能来自受控词表;不要用语料里的反馈词(会查询漂移) |
| 换前端协议 | `runtime/api/ui_stream.py` | part 序列有状态机(start/delta/end),不能只改字段名 |
| 接观测后端 | `runtime/obs/spans.py` | 保持 GenAI semconv 键名;导出交给 collector |

---

## 11. 已知取舍

1. **RRF 的 k 固定 60**。没做自适应;语料小的时候 k 影响很小,语料大了应该按候选集规模调。
2. **BM25 在内存**。重建索引是 O(全量);上万 chunk 后应换 OpenSearch/ES,接口已经在 `bm25.py` 里隔开。
3. **单图单进程**。`registry.py` 支持多图注册(按 `graph_key`),但调度、限流、副本间会话粘性都没做 ——
   会话在 checkpointer 里,所以副本无状态,这一点是留好的。
4. **没有子图/多 Agent**。DeepIntel 那套"主 Agent + 同步/异步子代理"适合复杂流程,
   当前场景(问答 + 单步业务操作)用不上,强行上只会增加状态同步成本。
5. **审计只记治理面操作**(登录、建项目、改模型、改用户)。业务动作的审计目前落在
   `tool_events` 与 JSON 日志里,要合规级审计还得单独接一张表。


---

## 12. 安全、契约与追踪(v0.2 新增)

### 12.1 三层 guardrail 的顺序不可交换

```
用户输入 ──▶ 输入闸门(阻塞) ──▶ 图执行 ──▶ StreamGuard(逐 token) ──▶ 输出闸门(收尾) ──▶ SSE
                    │                          │
                    └─ 拦截: 不拉 Catalog、   └─ 拦截/脱敏: 替换 answer,
                       不调模型、spent=0          status=blocked, 推 guardrail 事件

工具执行前 ──▶ 参数闸门(caller 注入/密钥) ──▶ MCP Server 的 scope + 角色校验
```

- **输入闸门必须阻塞**:并行跑会让模型先花 token、甚至先执行工具;企业场景里"没花钱"优先。
- **输出闸门是兜底,不是权限过滤**:租户隔离在召回前做;输出闸门负责把"过滤被绕过"变成
  可告警事件(跨租户金丝雀 `ZQ-OTHER-TENANT`)。
- **参数闸门防的是模型**:`caller` 由运行时注入,模型无权自带;参数里也不允许夹带密钥。
- 判定结果只回传 `rules` 与计数,不回传命中原文 —— 否则日志本身成为泄露渠道。
- 流式输出不能等收尾再查:半截密钥已经发出去了。`StreamGuard` 在风险前缀出现时扣住后续
  token(最多晚 8 字符),流结束再做一次整体校验;PII 脱敏只作用于最终 answer,不承诺逐 token。

### 12.2 MCP 工具契约

`ToolSpec` 同时服务四个消费者:MCP Server(生成 schema)、Graph(决定是否 HITL)、
Client(决定重试策略)、平台(`GET /v1/tools`)。契约字段:

| 字段 | 作用 |
|---|---|
| `annotations.readOnlyHint` | 客户端无需理解业务即可判断是否写操作 |
| `annotations.destructiveHint` | 显式 `False`;协议默认是 `True`,不写会被当成危险操作 |
| `annotations.idempotentHint` | 写操作靠 `idempotency_key` 去重,重复提交不产生两张单 |
| `output_schema` | FastMCP 的 structured output;客户端可校验返回结构 |
| `schema_hash` | `sha256(input + output)[:16]`,快照测试与前端版本判断用 |
| `contract_version` | 契约整体版本,破坏性变更时递增 |

### 12.3 trace 贯穿

```
x-trace-id / traceparent
        │
        ▼
RequestContextMiddleware ── trace_id_ctx ──▶ JSON 日志(每行都带)
        │
        ▼
AgentRunner.invoke_agent ──▶ LLMGateway.chat ──▶ tools_node.execute_tool
        │                        │                      │
        └─ gen_ai.usage.*        └─ gen_ai.request.model└─ gen_ai.tool.name
```

span 输出一条 JSON(见 `runtime/obs/spans.py`),属性名对齐 OTel GenAI semconv;
没有内置 exporter,是为了不把应用绑死在某个观测厂商。

### 12.4 评估门禁

`eval/run_eval.py` 产出 `var/eval_report.json`;`scripts/eval_gate.py` 以
`full_mq` 为基线,Recall@5 < 0.95 或 MRR < 0.85 直接让 CI 失败。
指标是确定性的(无 LLM 裁判),因此门禁不会因模型波动随机红灯。
