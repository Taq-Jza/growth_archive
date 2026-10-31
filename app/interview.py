"""面试追问预演（F-09）。

面试官看到一条简历条目，一定会往下问。这个功能提前把「会被问什么」和「你答得上来吗」
摆出来 —— 后者更重要：它反向暴露出记录写得不够细的地方。

诚实是硬约束：原文没写的细节必须标成「答不上来」并列入缺失项，不允许用常识补全。
"""
from __future__ import annotations

from . import db, llm
from .match import entity_ok

MAX_QUESTIONS = 6

STATUS_LABEL = {
    "can_answer": "能答",
    "shallow": "答得浅",
    "cannot": "答不上来",
}

INTERVIEW_SYS = """你是求职面试教练。基于候选人给出的一条经历记录，生成 5-6 个面试官最可能追问的问题。

只输出 JSON：
{"questions":[{"q":"面试问题","probe":"考察点（不超过15字）","status":"can_answer|shallow|cannot","points":["回答要点"],"missing":"记录里缺失但面试官会追问的信息"}]}

status 判定标准（必须严格按这个判）：
- can_answer：记录里有可以直接引用的事实来支撑回答（哪怕细节不全）。只要记录写了「做了什么」和「大致怎么做」，就算能答。
- shallow：记录只提了一句，只能泛泛而谈，需要临场发挥才能撑住。
- cannot：记录几乎没提这件事，候选人大概率答不上来。

注意：**不要因为「没有量化结果」就判成答不上来**，量化与否不影响能否回答。
同一份记录里，can_answer 和 cannot 通常都会出现，不要一边倒。

硬性约束：
1. points 只能取自记录原文，不得补充原文没有的事实、数字、奖项、时间。
2. 如果记录里没有量化结果，不要在 points 里编造数字。
3. missing 要具体写出缺什么（如「没写你在团队里负责哪一部分」），不要写「信息不足」这种空话。
4. 问题要像真实面试官问的，具体、有指向，不要问「请介绍一下这个项目」这种空泛问题。
5. 用中文。"""


def _normalize(raw: object, source: str) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw[:MAX_QUESTIONS]:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q", "")).strip()[:120]
        if len(q) < 6:
            continue
        status = item.get("status")
        if status not in STATUS_LABEL:
            status = "shallow"
        points: list[str] = []
        for p in item.get("points") or []:
            p = str(p).strip()[:120]
            # 回答要点里出现原文没有的数字/奖项 → 丢弃该条要点，避免教人编造
            if len(p) >= 4 and entity_ok(p, source):
                points.append(p)
        missing = str(item.get("missing", "")).strip()[:120]
        probe = str(item.get("probe", "")).strip()[:30]
        out.append(
            {
                "q": q,
                "probe": probe,
                "status": status,
                "status_label": STATUS_LABEL[status],
                "points": points[:4],
                "missing": missing,
            }
        )
    return out


def prepare(user_id: int, document_id: int) -> dict:
    doc = db.one(
        "SELECT id, title, body, kind FROM documents WHERE id = ? AND user_id = ?",
        (document_id, user_id),
    )
    if not doc:
        return {"error": "记录不存在。"}
    if not llm.available():
        return {"error": "模型服务不可用，暂时无法预演。记录本身不受影响。"}

    source = f"{doc['title']}\n{doc['body']}"
    messages = [
        {"role": "system", "content": INTERVIEW_SYS},
        {
            "role": "user",
            "content": f"标题：{doc['title']}\n类型：{doc['kind']}\n记录正文：\n{doc['body'][:6000]}",
        },
    ]

    questions: list[dict] = []
    for attempt in range(2):
        data = llm.chat_json(messages, timeout=40)
        raw = (data or {}).get("questions") if isinstance(data, dict) else data
        questions = _normalize(raw, source)
        if questions:
            break
        if attempt == 0:
            messages = messages + [
                {"role": "user", "content": "上次输出无法解析。请严格只输出 JSON，questions 至少 5 项。"}
            ]

    if not questions:
        return {"error": "预演失败：模型未返回可用结果，稍后可重试。"}

    stats = {
        "can_answer": sum(1 for q in questions if q["status"] == "can_answer"),
        "shallow": sum(1 for q in questions if q["status"] == "shallow"),
        "cannot": sum(1 for q in questions if q["status"] == "cannot"),
        "total": len(questions),
    }
    return {"doc": doc, "questions": questions, "stats": stats}
