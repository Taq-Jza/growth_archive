"""课程助手：问题分析（F-10）。

和「提问」不同，这里要的是结构化诊断：
先定位你的问题属于哪个知识点，再给解释，再指出你资料里缺什么。

关键约束：**每一句解释都必须挂上资料片段的编号**。
没有编号的句子，说明它是模型从课外知识里补的 —— 直接丢弃。
课程场景下「讲得漂亮但资料里没有」比「答不上来」危险得多。
"""
from __future__ import annotations

from . import llm, rag

ANALYZE_SYS = """你是这门课的助教。学生提出一个没弄懂的问题，你要基于**给定课程资料**做结构化分析。

只输出 JSON：
{"topic":"问题涉及的知识点（不超过15字）",
 "core":"一句话点出关键（不超过40字）",
 "core_ref":1,
 "explain":[{"step":"一句话解释","ref":1}],
 "related":[1,2],
 "missing":"课程资料里没有覆盖到的部分；资料足够回答时填空字符串"}

硬性约束（最重要）：
1. **每一句 core 和 explain 都必须能在给定片段里找到依据，并在 core_ref / ref 里填上对应片段编号。**
2. 如果给定片段里没有直接讲到这个问题，就不要用你自己的知识补充 ——
   把 core 写成「课程资料里没有直接讲到这个问题的答案」，explain 只复述片段里确实提到的相关内容。
3. ref 必须是真实存在的片段编号（1 到 N），不许编造编号。
4. 不要编造公式、数字、定理名、人名。
5. 用中文，语气像助教，不要客套话。"""


def _valid_ref(value, upper: int) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= upper else None


def analyze(user_id: int, question: str) -> dict:
    question = (question or "").strip()
    if len(question) < 4:
        return {"error": "问题太短了，多写几个字。"}
    if not llm.available():
        return {"error": "模型服务不可用，暂时无法分析。"}

    chunks, mode = rag.retrieve(user_id, question, kind="course")
    chunks = chunks[:4]
    if not chunks:
        return {
            "error": "课程资料里没有找到相关内容。先上传这门课的课件或笔记，再来问会更准。"
        }

    context = "\n\n".join(
        f"[{i}] 《{c['title']}》\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    messages = [
        {"role": "system", "content": ANALYZE_SYS},
        {
            "role": "user",
            "content": f"我的问题：{question}\n\n【课程资料片段】\n{context}\n\n请只依据以上片段分析。",
        },
    ]

    data = llm.chat_json(messages, timeout=40) or {}
    if not isinstance(data, dict) or not data.get("core"):
        return {"error": "分析失败，模型没有返回可用结果。稍后再试一次。"}

    upper = len(chunks)

    # 只保留能被资料支撑的解释步骤
    steps: list[dict] = []
    for item in data.get("explain") or []:
        if isinstance(item, dict):
            text = str(item.get("step", "")).strip()
            ref = _valid_ref(item.get("ref"), upper)
        else:
            text, ref = str(item).strip(), None
        if len(text) < 4:
            continue
        if ref is None:
            # 没有有效编号 = 这句话不是从资料里来的，丢掉
            continue
        steps.append({"step": text[:200], "ref": ref})

    core_ref = _valid_ref(data.get("core_ref"), upper)
    dropped = len([x for x in (data.get("explain") or [])]) - len(steps)

    if not steps and core_ref is None:
        return {
            "error": "你的课程资料里没有能支撑这个问题的内容。"
            "补一份相关课件或笔记之后再问，答案才不会掺进课外知识。"
        }

    related_ids: list[int] = []
    for n in [core_ref] + [s["ref"] for s in steps] + [
        _valid_ref(x, upper) for x in (data.get("related") or [])
    ]:
        if n and n not in related_ids:
            related_ids.append(n)

    citations = [
        {
            "n": n,
            "doc_id": chunks[n - 1]["document_id"],
            "chunk_id": chunks[n - 1]["chunk_id"],
            "title": chunks[n - 1]["title"],
        }
        for n in related_ids
    ]

    return {
        "question": question,
        "topic": str(data.get("topic", "")).strip()[:30],
        "core": str(data.get("core", "")).strip()[:80],
        "explain": steps,
        "missing": str(data.get("missing", "")).strip()[:160],
        "citations": citations,
        "dropped": dropped,
        "mode": mode,
        "chunks": upper,
    }
