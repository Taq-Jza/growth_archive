"""D1 第一步：验证模型接口能不能用。先跑这个，再写业务代码。

用法：
    python scripts/check_llm.py
    python scripts/check_llm.py "用一句话解释什么是 RAG"
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, llm  # noqa: E402


def mask(key: str) -> str:
    if not key:
        return "（未配置）"
    return key[:6] + "…" + key[-4:] if len(key) > 12 else "***"


def main() -> int:
    print("=" * 52)
    print(f"BASE_URL   {config.LLM_BASE_URL}")
    print(f"API_KEY    {mask(config.LLM_API_KEY)}")
    print(f"CHAT_MODEL {config.CHAT_MODEL}")
    print(f"TIMEOUT    {config.LLM_TIMEOUT}s")
    print("=" * 52)

    if not llm.available():
        print("[X] 未配置 LLM_API_KEY，请在 .env 中填写后重试。")
        return 1

    question = sys.argv[1] if len(sys.argv) > 1 else "用一句话说明什么是检索增强生成。"

    print("\n[1/2] 非流式调用…")
    t0 = time.time()
    try:
        out = llm.chat([{"role": "user", "content": question}], max_tokens=120)
    except llm.LLMUnavailable as exc:
        print(f"[X] 失败：{exc}")
        print("    排查：key 是否正确 / 模型名是否存在 / 账户是否欠费 / 网络能否出站")
        return 2
    dt = time.time() - t0
    print(f"    耗时 {dt:.2f}s，返回 {len(out)} 字")
    print(f"    内容：{out[:120]}")

    print("\n[2/2] 流式调用（看首字延迟）…")
    t0 = time.time()
    first = None
    total = 0
    try:
        for token in llm.chat_stream(
            [{"role": "user", "content": "列出三个学习方法的名称，每个不超过六个字。"}],
            max_tokens=80,
        ):
            if first is None:
                first = time.time() - t0
            total += len(token)
    except llm.LLMUnavailable as exc:
        print(f"[X] 流式失败：{exc}")
        return 3

    print(f"    首字延迟 {first:.2f}s，共 {total} 字")
    print("\n结论：")
    print(f"    非流式 {dt:.2f}s / 首字 {first:.2f}s")
    if first and first > 3:
        print("    [!] 首字超过 3 秒，演示会明显卡顿，建议换更小的模型档位。")
    else:
        print("    [OK] 可用于演示。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
