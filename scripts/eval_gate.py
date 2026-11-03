"""评估门禁：把"检索质量不能倒退"变成 CI 的硬失败。

阈值不是拍脑袋：当前基线是 Recall@5=1.000、MRR=0.912（见 README），
门禁留出 5 个点的余量，允许正常波动，但拦住"改完检索反而更差"的合并。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "var" / "eval_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评估门禁")
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--mode", default="full_mq", help="以哪条管线作为门禁基线")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-mrr", type=float, default=0.85)
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"找不到评估报告：{path}，请先运行 python -m eval.run_eval")
        return 2

    report = json.loads(path.read_text(encoding="utf-8"))
    modes = report.get("modes", {})
    if args.mode not in modes:
        print(f"报告里没有管线 {args.mode}，可选：{sorted(modes)}")
        return 2

    mode = modes[args.mode]
    checks = [
        ("Recall@5", mode["recall_at_5"], args.min_recall),
        ("MRR", mode["mrr"], args.min_mrr),
    ]
    failed = False
    for name, value, threshold in checks:
        status = "PASS" if value >= threshold else "FAIL"
        failed |= value < threshold
        print(f"{status}  {name}={value:.4f}  (>= {threshold:.4f})")

    print(f"基线：{args.mode} | 语料 {report['corpus']['chunks']} chunks | 评估集 {report['dataset_size']} 条")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
