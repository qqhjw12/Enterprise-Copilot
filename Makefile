.PHONY: help install infra up down logs test eval gate smoke verify-dates seed lint clean dev-api dev-rt dev-mcp dev-web

help:
	@echo "install  安装依赖"
	@echo "infra    仅启动 postgres/redis/qdrant"
	@echo "up       全量构建并启动"
	@echo "down     停止并清理容器"
	@echo "seed     灌入示例知识库 + 演示账号"
	@echo "test     跑单元测试（零 API Key）"
	@echo "eval     跑 RAG 评估（五路对比 / nDCG / 置信区间）"
	@echo "gate     评估门禁（Recall@5 / MRR 不倒退）"
	@echo "smoke    端到端 8 场景冒烟（需先起 8000/8001）"
	@echo "verify-dates 请假日期理解：多种写法 + 多轮补全"
	@echo "dev-api  本地起控制面 (8000)"
	@echo "dev-rt   本地起执行面 (8001)"
	@echo "dev-mcp  本地起 MCP Server (http 8765)"
	@echo "dev-web  本地起工作台 (8501)"

install:
	pip install -r requirements.txt

infra:
	docker compose up -d postgres redis

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

seed:
	python -m runtime.rag.ingest --tenant demo --path data/knowledge

test:
	pytest

eval:
	python -m eval.run_eval

smoke:
	python scripts/smoke.py

verify-dates:
	python scripts/verify_dates.py

dev-api:
	python -m uvicorn platform_api.main:app --reload --port 8000

# MCP 走 stdio 子进程：省掉单独起一个 Server 容器，工具仍是走完整 MCP 协议
dev-rt:
	MCP_OPS_COMMAND=python MCP_OPS_ARGS="-m,runtime.mcp_layer.server,--transport,stdio" \
	python -m uvicorn runtime.main:app --reload --port 8001

dev-mcp:
	python -m runtime.mcp_layer.server --transport http --port 8765

dev-web:
	COPILOT_API_BASE=http://localhost:8001 streamlit run web/app.py

lint:
	ruff check .

clean:
	rm -rf var .pytest_cache **/__pycache__
