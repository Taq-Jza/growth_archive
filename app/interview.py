"""面试追问预演（F-09）。

面试官看到一条简历条目，一定会往下问。这个功能提前把「会被问什么」和「你答得上来吗」
摆出来 —— 后者更重要：它反向暴露出记录写得不够细的地方。

诚实是硬约束：原文没写的细节必须标成「答不上来」并列入缺失项，不允许用常识补全。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

from . import config, db, llm, websearch
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


def _body_hash(body: str) -> str:
    return hashlib.sha1((body or "").encode("utf-8")).hexdigest()[:16]


def load_cached(user_id: int, doc) -> dict | None:
    """取上一次的预演结果。记录改过（正文 hash 不符）就当没有。"""
    row = db.one(
        "SELECT body_hash, data FROM interview_cache WHERE document_id = ? AND user_id = ?",
        (doc["id"], user_id),
    )
    if not row or row["body_hash"] != _body_hash(doc["body"]):
        return None
    try:
        payload = json.loads(row["data"])
    except Exception:  # noqa: BLE001 存坏了就当没有，不影响重新生成
        return None
    return payload or None


def _save_cache(user_id: int, doc, payload: dict) -> None:
    # doc 是 sqlite3.Row，不能直接 json 序列化 —— 存之前剔掉，读出来由调用方补回。
    # 这一点没处理时缓存永远写不进去，而异常被下面的 try 吞了，表现是「缓存一直不命中」。
    body = {k: v for k, v in payload.items() if k != "doc"}
    db.execute(
        """INSERT INTO interview_cache (document_id, user_id, body_hash, data, created_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(document_id) DO UPDATE SET
             user_id=excluded.user_id, body_hash=excluded.body_hash,
             data=excluded.data, created_at=excluded.created_at""",
        (doc["id"], user_id, _body_hash(doc["body"]), json.dumps(body, ensure_ascii=False),
         dt.datetime.now().isoformat(timespec="seconds")),
    )


def prepare(user_id: int, document_id: int, force: bool = False) -> dict:
    """生成预演题目。

    默认先看缓存：这条记录的正文没改过就直接用上次的题，
    省掉 20 秒等待和一次检索调用。force=True 才重新生成（页面上的「换一批」）。
    """
    doc = db.one(
        "SELECT id, title, body, kind FROM documents WHERE id = ? AND user_id = ?",
        (document_id, user_id),
    )
    if not doc:
        return {"error": "记录不存在。"}

    if not force:
        cached = load_cached(user_id, doc)
        if cached:
            cached["doc"] = doc          # 缓存里没存，这里补回，模板结构保持一致
            cached["cached"] = True
            return cached

    if not llm.available():
        return {"error": "模型服务不可用，暂时无法预演。记录本身不受影响。"}

    source = f"{doc['title']}\n{doc['body']}"

    # 先查一遍近期真题。查不到就是空列表，正文里也不会多出一段空话。
    refs = websearch.search(f"{doc['title'][:24]} 面试 常见问题 2025", count=6) if websearch.available() else []
    ref_block = ""
    if refs:
        lines = "\n".join(f"- {r['snippet']}" for r in refs if r.get("snippet"))
        if lines:
            ref_block = (
                "\n\n以下是近期公开面试题的检索结果。**只能用来决定「问什么方向」**：\n"
                "不要把这些内容当成本条记录里发生过的事实，也不要照抄原题，"
                "问题仍要对准上面这条记录。\n" + lines
            )

    messages = [
        {"role": "system", "content": INTERVIEW_SYS},
        {
            "role": "user",
            "content": f"标题：{doc['title']}\n类型：{doc['kind']}\n记录正文：\n{doc['body'][:6000]}{ref_block}",
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
    payload = {
        "doc": doc,
        "questions": questions,
        "stats": stats,
        "web": {"count": len(refs), "provider": config.SEARCH_PROVIDER} if refs else None,
        "cached": False,
    }
    try:
        _save_cache(user_id, doc, payload)
    except Exception:  # noqa: BLE001 存缓存失败不该让这次结果作废
        pass
    return payload
