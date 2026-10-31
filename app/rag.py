"""检索与问答链路（对应《设计方案》D-02 / D-07 / D-11 的证据组织部分）。

两阶段检索：关键词召回 Top20 → 模型重排 Top5。
三级降级：无 key → 直接返回原文片段；重排失败 → 用召回顺序；生成失败 → 返回原文片段。
"""
from __future__ import annotations

import re
from typing import Iterator

from . import config, llm
from .search import keyword_search, rrf_fuse, semantic_search

MAX_CANDIDATES = 20
TOP_K = 5

ANSWER_SYS = """你是用户个人成长档案的助手。严格遵守：
1. 只能依据下面提供的记录片段回答，禁止使用你自己的知识，禁止编造经历、数字、奖项、时间。
2. 每个结论后面用 [编号] 标注依据，编号只能取自给定片段，即 [1] 到 [N]。
3. 若片段不足以回答问题，直接回答「你的档案里没有相关记录」，不要猜测、不要给建议性的编造。
4. 用中文回答，简洁，控制在 200 字以内，不要写标题。"""

RERANK_SYS = """你是检索结果重排器。用户会给出一个问题和若干编号候选片段。
只输出一个 JSON 数组，元素是候选编号（从 1 开始），按与问题的相关程度从高到低排列，最多 5 个。
不要输出任何解释、不要输出片段内容本身。示例：[3,1,7,2,5]"""


def invalid_markers(text: str, n: int) -> list[str]:
    """找出超出 [1, n] 范围的引用标号。"""
    found = re.findall(r"\[(\d{1,2})\]", text or "")
    return [m for m in found if not (1 <= int(m) <= n)]


def rerank(question: str, candidates: list[dict], top_k: int = TOP_K) -> list[dict]:
    """模型重排。任何失败都退回召回顺序，绝不阻断问答。"""
    if len(candidates) <= top_k:
        return candidates
    if not llm.available():
        return candidates[:top_k]

    listing = "\n".join(
        f"{i}. {c['text'][:160].replace(chr(10), ' ')}" for i, c in enumerate(candidates, 1)
    )
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": RERANK_SYS},
                {"role": "user", "content": f"问题：{question}\n\n候选片段：\n{listing}"},
            ],
            temperature=0.0,
            max_tokens=60,
            timeout=5,
        )
    except llm.LLMUnavailable:
        return candidates[:top_k]

    picked: list[dict] = []
    seen: set[int] = set()
    for num in re.findall(r"\d+", raw or ""):
        idx = int(num)
        if 1 <= idx <= len(candidates) and idx not in seen:
            picked.append(candidates[idx - 1])
            seen.add(idx)
        if len(picked) >= top_k:
            break
    return picked or candidates[:top_k]


def retrieve(user_id: int, question: str, kind: str | None = None) -> tuple[list[dict], str]:
    """混合检索：关键词路与语义路并行召回，RRF 融合后再交给模型重排。

    kind 不为空时把检索范围限定在该分类（课程助手就靠这个只看课程资料）。
    返回 (片段列表, 模式)。模式用于前端展示与验收取证：
    hybrid = 双路融合，keyword = 语义路不可用时的降级。
    """
    keyword_hits = keyword_search(user_id, question, limit=MAX_CANDIDATES, kind=kind)
    semantic_hits = semantic_search(user_id, question, limit=MAX_CANDIDATES, kind=kind)

    if semantic_hits:
        merged = rrf_fuse([keyword_hits, semantic_hits], limit=MAX_CANDIDATES)
        mode = "hybrid"
    else:
        merged = keyword_hits
        mode = "keyword"

    return rerank(question, merged, TOP_K), mode


def to_citations(chunks: list[dict]) -> list[dict]:
    return [
        {
            "n": i,
            "doc_id": c["document_id"],
            "chunk_id": c["chunk_id"],
            "title": c["title"],
            "text": c["text"],
        }
        for i, c in enumerate(chunks, 1)
    ]


def _fallback_body(chunks: list[dict], reason: str) -> str:
    lines = [
        f"{reason}",
        "",
        "以下是你档案中最相关的片段，可直接点开核对：",
        "",
    ]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] {c['text'][:180].strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def _messages(question: str, chunks: list[dict]) -> list[dict]:
    ctx = "\n\n".join(
        f"[{i}] 《{c['title']}》\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    return [
        {"role": "system", "content": ANSWER_SYS},
        {
            "role": "user",
            "content": f"我的问题：{question}\n\n【记录片段】\n{ctx}\n\n请只依据以上片段回答。",
        },
    ]


def answer_stream(user_id: int, question: str, top_k: int = TOP_K, kind: str | None = None) -> Iterator[tuple[str, dict]]:
    """产出 (event, payload)：token / replace / done / fail。"""
    question = (question or "").strip()
    if not question:
        yield ("done", {"answer": "请输入你的问题。", "citations": [], "degraded": False, "mode": "keyword"})
        return

    chunks, mode = retrieve(user_id, question, kind=kind)
    chunks = chunks[:top_k]
    citations = to_citations(chunks)

    # 无依据：明确拒答，不编造
    if not chunks:
        yield (
            "done",
            {
                "answer": "你的档案里没有相关记录。可以换个说法，或先补充一条相关经历。",
                "citations": [],
                "degraded": False,
                "mode": mode,
            },
        )
        return

    # L2 降级：模型不可用，直接给原文
    if not llm.available():
        yield (
            "done",
            {
                "answer": _fallback_body(chunks, "模型服务暂不可用，已为你切换到检索结果。"),
                "citations": citations,
                "degraded": True,
                "mode": mode,
            },
        )
        return

    messages = _messages(question, chunks)
    buf: list[str] = []
    try:
        for token in llm.chat_stream(messages, timeout=config.LLM_TIMEOUT):
            buf.append(token)
            yield ("token", {"text": token})
    except llm.LLMUnavailable as exc:
        yield ("replace", {"text": _fallback_body(chunks, f"模型服务暂时不可用（{str(exc)[:60]}）。")})
        yield ("done", {"citations": citations, "degraded": True, "mode": mode})
        return

    text = "".join(buf).strip()

    # 引用校验：出现越界标号则重试一次
    if invalid_markers(text, len(chunks)):
        try:
            fixed = llm.chat(
                messages
                + [{"role": "user", "content": f"只允许使用 [1] 到 [{len(chunks)}] 的编号，请重写回答。"}],
                temperature=0.0,
                timeout=config.LLM_TIMEOUT,
            )
            if fixed and not invalid_markers(fixed, len(chunks)):
                yield ("replace", {"text": fixed.strip()})
        except llm.LLMUnavailable:
            pass

    yield ("done", {"citations": citations, "degraded": False, "mode": mode})
