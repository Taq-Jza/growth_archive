"""知识图谱：把某个标签下的记录按章节组织成可点击的图。

做法分两步：
1. 抽取（build）：让模型给每条记录判断「属于哪一章」并抽出关键概念，结果落库；
2. 布局（layout）：纯几何计算，把章节放在内圈、记录放在外圈，模板只负责画。

为什么不让模型直接生成图：坐标、半径、重叠这些事模型算不准，
而且每次生成结果都不一样；固定布局至少让同一份数据看起来是同一张图。
"""
from __future__ import annotations

import json
import math

from . import db, llm

EXTRACT_PROMPT = """你在帮学生整理一门课的知识结构。
下面是一条课程笔记的标题与正文，请判断它属于哪一章，并抽取 3–5 个关键概念。

要求：
1. chapter 只写**主题名**，不超过 12 字，例如「索引与查询优化」「进程与线程」「假设检验」。
   不要写「第 N 章」—— 不同课程的章号会撞，图上就分不清。
   如果两条笔记讲的是同一主题，chapter 必须完全一致，不要每次造新名字。
2. concepts 是正文里真正出现过的名词（方法、定理、术语），不要写整句。
3. 只输出 JSON：{"chapter":"...","concepts":["...","..."]}"""


def _docs_in_tag(user_id: int, kind: str | None, tag: str | None) -> list:
    """取记录：给了 tag 就按主标签筛，没给就是该分类的全部。

    口径与分类页、标签收藏夹一致 —— 否则图上的条数和列表里的对不上。
    """
    from .ingest import KINDS

    labels = tuple(KINDS.values())
    ph = ",".join("?" for _ in labels)
    extra = " AND d.kind = ?" if kind else ""
    params: list = [user_id]
    if kind:
        params.append(kind)

    tag_sql = ""
    if tag:
        tag_sql = f""" AND (SELECT t2.name FROM document_tags dt2 JOIN tags t2 ON t2.id = dt2.tag_id
                            WHERE dt2.document_id = d.id AND t2.name NOT IN ({ph})
                            ORDER BY dt2.tag_id LIMIT 1) = ?"""
        params.extend(labels)
        params.append(tag)

    return db.rows(
        f"""SELECT d.id, d.title, d.body
            FROM documents d
            WHERE d.user_id = ?{extra}{tag_sql}
            ORDER BY COALESCE(d.period, d.created_at)""",
        tuple(params),
    )


def build(user_id: int, kind: str | None, tag: str | None, force: bool = False) -> dict:
    """为标签下的每条记录抽取章节与概念。已抽过的不重复调用模型。"""
    docs = _docs_in_tag(user_id, kind, tag)
    done = {r["document_id"] for r in db.rows("SELECT document_id FROM doc_kg")}
    targets = [d for d in docs if force or d["id"] not in done]
    ok, failed = 0, 0

    for d in targets:
        data = llm.chat_json(
            [
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"标题：{d['title']}\n\n正文：\n{(d['body'] or '')[:2500]}"},
            ],
            timeout=40,
        )
        if not isinstance(data, dict) or not data.get("chapter"):
            failed += 1
            continue
        chapter = str(data.get("chapter"))[:20]
        concepts = [str(c)[:16] for c in (data.get("concepts") or [])][:5]
        with db.get_conn() as conn:
            conn.execute(
                """INSERT INTO doc_kg (document_id, chapter, concepts, created_at)
                   VALUES (?,?,?,datetime('now'))
                   ON CONFLICT(document_id) DO UPDATE SET chapter=excluded.chapter, concepts=excluded.concepts""",
                (d["id"], chapter, json.dumps(concepts, ensure_ascii=False)),
            )
        ok += 1
    return {"ok": ok, "failed": failed, "total": len(docs)}


def load(user_id: int, kind: str | None, tag: str | None) -> list[dict]:
    docs = _docs_in_tag(user_id, kind, tag)
    meta = {r["document_id"]: r for r in db.rows("SELECT * FROM doc_kg")}
    out = []
    for d in docs:
        m = meta.get(d["id"])
        if not m:
            continue
        out.append(
            {
                "id": d["id"],
                "title": d["title"],
                "chapter": m["chapter"] or "未分章",
                "concepts": json.loads(m["concepts"] or "[]"),
            }
        )
    return out


def _label(x: float, y: float, cx: float, cy: float, pad: float = 14.0) -> tuple:
    """标签放在节点外侧，并按所在半圆决定对齐方式，避免压住连线。"""
    dx, dy = x - cx, y - cy
    dist = math.hypot(dx, dy) or 1.0
    lx, ly = x + dx / dist * pad, y + dy / dist * pad
    anchor = "middle"
    if dx > 10:
        anchor = "start"
    elif dx < -10:
        anchor = "end"
    return round(lx, 1), round(ly + 3.5, 1), anchor


def layout(rows: list[dict], width: int = 860, height: int = 560) -> dict:
    """中心 = 标签；内圈 = 章节；外圈 = 记录。

    章节按记录数从多到少排，从 12 点方向顺时针铺开，
    同一章的记录聚在它周围的一小段角度里，便于「看出一章包含哪些笔记」。
    """
    cx, cy = width / 2, height / 2
    r1, r2 = 132, 236

    chapters: dict[str, list[dict]] = {}
    for r in rows:
        chapters.setdefault(r["chapter"], []).append(r)

    ordered = sorted(chapters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    nodes: list[dict] = [{"type": "center", "label": "", "x": cx, "y": cy}]
    edges: list[dict] = []

    n = max(1, len(ordered))
    for ci, (name, docs) in enumerate(ordered):
        ang = -math.pi / 2 + 2 * math.pi * ci / n
        chx, chy = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
        lx, ly, anchor = _label(chx, chy, cx, cy, 18)
        nodes.append(
            {"type": "chapter", "label": name, "x": round(chx, 1), "y": round(chy, 1),
             "count": len(docs), "lx": lx, "ly": ly, "anchor": anchor}
        )
        edges.append({"x1": cx, "y1": cy, "x2": chx, "y2": chy, "kind": "spine"})

        m = len(docs)
        # 记录铺开的角度：条目越多铺得越开，但最多 0.9 弧度，避免和外圈重叠
        spread = min(0.9, 0.26 * max(1, m - 1))
        for di, doc in enumerate(docs):
            off = 0 if m == 1 else (di / (m - 1) - 0.5) * spread
            a = ang + off
            x, y = cx + r2 * math.cos(a), cy + r2 * math.sin(a)
            lx, ly, anchor = _label(x, y, cx, cy, 16)
            nodes.append(
                {
                    "type": "doc",
                    "label": doc["title"][:14],
                    "full": doc["title"],
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "doc_id": doc["id"],
                    "concepts": doc["concepts"],
                    "lx": lx, "ly": ly, "anchor": anchor,
                }
            )
            edges.append({"x1": chx, "y1": chy, "x2": x, "y2": y, "kind": "leaf"})

    return {"nodes": nodes, "edges": edges, "width": width, "height": height}
