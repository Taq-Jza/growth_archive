"""思维导图：把一份资料的知识结构抽成树，再算成能画的坐标。

和 kg.py 的分工：
- 知识图谱看「多条记录之间」的章节关系，是跨记录的；
- 思维导图看「一份资料内部」的结构，是单条记录的。

两者都用模型抽结构、本地算坐标，理由一样：坐标与重叠模型算不准，
每次生成还不一样；固定布局至少保证同一份数据看起来是同一张图。
"""
from __future__ import annotations

import json

from . import db, llm

PROMPT = """你在帮学生把一份课程资料整理成思维导图。
请从下面的正文里提取层级结构，只输出 JSON，不要解释：

{"title": "整份资料的主题，不超过 14 字",
 "children": [
   {"title": "一级主题，不超过 12 字", "children": [
       {"title": "要点，不超过 20 字"},
       {"title": "要点，不超过 20 字"}
   ]}
 ]}

硬性要求：
1. 一级主题 3–8 个，按正文出现顺序排列，覆盖主要章节或题型；
2. 每个一级主题下 2–6 个要点，写知识点本身，不写整句、不写「本段介绍了」；
3. 一级主题和要点都必须来自正文，不得补充正文没有的内容；
4. 数字、公式符号、专有名词原样保留（如「u=100 m/s」「Δφ=4π」「p 值」）；
5. 只做两层，不要出现第三层。"""

MAX_CHARS = 12000


def clean(node, depth: int = 0) -> dict | None:
    """把模型返回的节点洗成固定形状：title 必有，children 最多两层。"""
    if not isinstance(node, dict):
        return None
    title = str(node.get("title") or "").strip()
    if not title:
        return None
    item = {"title": title[:28]}
    if depth < 2:
        kids = []
        for c in (node.get("children") or [])[:8]:
            got = clean(c, depth + 1)
            if got:
                kids.append(got)
        if kids:
            item["children"] = kids
    return item


def build(user_id: int, doc_id: int, force: bool = False) -> dict | None:
    """为一条记录生成思维导图。已生成过且不强制时直接返回旧的。"""
    doc = db.one(
        "SELECT id, title, body FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)
    )
    if not doc:
        return None

    if not force:
        row = db.one("SELECT data FROM doc_map WHERE document_id = ?", (doc_id,))
        if row and row["data"] and row["data"] != "{}":
            return load(doc_id)

    data = llm.chat_json(
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"标题：{doc['title']}\n\n正文：\n{(doc['body'] or '')[:MAX_CHARS]}"},
        ],
        timeout=60,
    )
    root = clean(data) if isinstance(data, dict) else None
    if not root:
        return None
    root["title"] = root["title"] or doc["title"]

    # 中心节点固定用资料标题，模型给的主题名常比标题长，图会被撑歪
    root_title = doc["title"][:18]
    if root.get("title") and len(root["title"]) <= 18:
        root_title = root["title"]
    root["title"] = root_title

    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO doc_map (document_id, data, created_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(document_id) DO UPDATE SET data=excluded.data, created_at=excluded.created_at""",
            (doc_id, json.dumps(root, ensure_ascii=False)),
        )
    return load(doc_id)


def load(doc_id: int) -> dict | None:
    row = db.one("SELECT data FROM doc_map WHERE document_id = ?", (doc_id,))
    if not row or not row["data"] or row["data"] == "{}":
        return None
    try:
        return json.loads(row["data"])
    except json.JSONDecodeError:
        return None


def _leaves(node: dict) -> int:
    kids = node.get("children") or []
    if not kids:
        return 1
    return sum(_leaves(k) for k in kids)


def layout(
    root: dict,
    col_w: int = 208,
    row_h: int = 34,
    gap: int = 10,
    pad_left: int = 12,
) -> dict:
    """把树排成横向图：根在左，一级主题一列，要点再一列。

    纵向位置按「叶子数」分配：一个节点占据它所有后代的连续区间，
    父节点居中对齐到子节点区间的中点 —— 这样连线不会互相穿插。
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    total = _leaves(root)
    height = max(1, total) * row_h + (max(1, total) - 1) * gap
    pad_top = row_h / 2

    def walk(node: dict, depth: int, y0: float) -> float:
        """返回这个子树占据的中心 y。"""
        kids = node.get("children") or []
        own_h = _leaves(node) * row_h + (_leaves(node) - 1) * gap
        if not kids:
            cy = y0 + row_h / 2
        else:
            cy = y0 + own_h / 2
            cursor = y0
            for k in kids:
                ky = walk(k, depth + 1, cursor)
                edges.append(
                    {
                        "x1": pad_left + depth * col_w + 160,
                        "y1": cy,
                        "x2": pad_left + (depth + 1) * col_w,
                        "y2": ky,
                    }
                )
                cursor += _leaves(k) * row_h + (_leaves(k) - 1) * gap
        nodes.append(
            {
                "x": pad_left + depth * col_w,
                "y": round(cy, 1),
                "title": node["title"],
                "depth": depth,
                "leaf": not kids,
            }
        )
        return cy

    walk(root, 0, pad_top)

    width = max(n["x"] for n in nodes) + 190
    return {
        "nodes": nodes,
        "edges": edges,
        "width": int(width),
        "height": int(height + row_h),
    }
