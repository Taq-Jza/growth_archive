"""要点提炼（F-06，对应设计方案 D-08）。

把一条成长记录压成 1-6 张要点卡：做了什么 / 结果如何 / 证明了什么能力 / 还可补充什么。
这套卡片有两个用途：一是回顾，二是作为「岗位匹配」写简历条目时的中间层。

三条不可妥协的约束：
1. 只依据原文，不得引入原文没有的事实；
2. 结果里的数字、奖项、时间必须在原文中出现，否则整段替换成固定文案；
3. 同一记录重复提炼时覆盖旧卡片，不追加。
"""
from __future__ import annotations

import datetime as dt

from . import db, llm
from .match import entity_ok

MAX_CARDS = 6
NO_QUANT = "原文未给出量化结果"

CARD_SYS = """你是经历提炼助手。把一段成长记录压成 3-4 张要点卡，用于回顾和写简历。
只输出 JSON：{"cards":[{"what":"做了什么，不超过40字","result":"结果如何，不超过40字","ability":"证明了什么能力，不超过20字","todo":"还可补充什么，不超过20字"}]}

结构要求：
- 第 1 张必须是「成果卡」：what 概括这件事本身，result 写最终成果。
  原文只要提到奖项、名次、成绩、量化结果，result 必须原样写出（如「获省赛二等奖」「准确率 91%」）。
- 第 2 张起是「要点卡」：分别讲做法、难点、收获，一张只讲一件事。

硬性约束：
1. 只能使用原文中出现的信息，不得补充原文没有的事实、单位、时间。
2. result 中若写数字、比例、奖项、时间，必须与原文完全一致，沿用原文写法。
3. 原文确实没有量化结果时，result 填「原文未给出量化结果」，禁止编造。
4. 用陈述句，不用第一人称，不加序号。
5. 不要把同一件事拆成多张流程卡（「准备过程」「刷题方式」这类应合并为一张），
   更不要为了凑数量而拆分——3 张有信息量的卡好过 6 张流水账。"""


def _normalize(raw: object, source: str) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw[:MAX_CARDS]:
        if not isinstance(item, dict):
            continue
        what = str(item.get("what", "")).strip()[:80]
        if len(what) < 4:
            continue
        result = str(item.get("result", "")).strip()[:80]
        ability = str(item.get("ability", "")).strip()[:40]
        todo = str(item.get("todo", "")).strip()[:40]

        # what / ability / todo 出现原文没有的实体 → 整张卡丢弃（宁可少一张，不留假话）
        if not entity_ok(what, source) or not entity_ok(ability, source) or not entity_ok(todo, source):
            continue
        # result 出现原文没有的数字/奖项 → 降级为固定文案，保留卡片
        if result and not entity_ok(result, source):
            result = NO_QUANT
        if not result:
            result = NO_QUANT

        out.append({"what": what, "result": result, "ability": ability, "todo": todo})
    return out


def extract(user_id: int, document_id: int) -> dict:
    doc = db.one(
        "SELECT id, title, body, kind FROM documents WHERE id = ? AND user_id = ?",
        (document_id, user_id),
    )
    if not doc:
        return {"error": "记录不存在。"}
    if not llm.available():
        return {"error": "模型服务不可用，暂时无法提炼。记录本身不受影响。"}

    source = doc["body"]
    # 实体校验要连标题一起比：正文常写「拿了省二」，而「省赛二等奖」在标题里。
    # 只比对正文会把模型正确写出的奖项误判为编造。
    source_for_check = f"{doc['title']}\n{doc['body']}"
    messages = [
        {"role": "system", "content": CARD_SYS},
        {"role": "user", "content": f"标题：{doc['title']}\n类型：{doc['kind']}\n正文：\n{source[:6000]}"},
    ]

    cards: list[dict] = []
    for attempt in range(2):  # 首次失败重试一次
        data = llm.chat_json(messages, timeout=30)
        raw = (data or {}).get("cards") if isinstance(data, dict) else data
        cards = _normalize(raw, source_for_check)
        if cards:
            break
        if attempt == 0:
            messages = messages + [
                {"role": "user", "content": "上次输出无法解析或为空。请严格只输出 JSON，且 cards 至少一项。"}
            ]

    if not cards:
        return {"error": "提炼失败：模型未返回可用的卡片。已保留原文，可稍后重试。"}

    now = dt.datetime.now().isoformat(timespec="seconds")
    with db.get_conn() as conn:
        conn.execute("DELETE FROM cards WHERE document_id = ? AND user_id = ?", (document_id, user_id))
        for c in cards:
            conn.execute(
                """INSERT INTO cards (user_id, document_id, what, result, ability, todo, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, document_id, c["what"], c["result"], c["ability"], c["todo"], now),
            )

    return {"cards": cards, "count": len(cards)}
