"""思维导图：把一份资料的知识结构抽成树，再算成能画的坐标。

和 kg.py 的分工：
- 知识图谱看「多条记录之间」的章节关系，是跨记录的；
- 思维导图看「一份资料内部」的结构，是单条记录的。

两者都用模型抽结构、本地算坐标，理由一样：坐标与重叠模型算不准，
每次生成还不一样；固定布局至少保证同一份数据看起来是同一张图。

节点不是章节名。第一版提示词写「一级主题 / 要点」，模型直接抄了小标题，
图上就只剩目录：`简谐振动`、`静电场`、`恒定磁场` —— 看名字和翻目录没区别。
现在每个节点都要带一句 `detail` 说明它到底讲什么，叶子节点还带 `quote`
（正文里逐字摘出的片段），点击能跳回原文那一段。
"""
from __future__ import annotations

import json

from . import db, llm

PROMPT = """你在帮学生把一份课程资料整理成思维导图。

这不是做目录。每个节点都要写清「这一条讲的是什么知识」，
不能只抄章节名或小标题 —— 那样的图和翻目录没有区别。

只输出 JSON：
{"title": "整份资料的主题，不超过 16 字",
 "children": [
   {"title": "一级主题，不超过 14 字",
    "detail": "这个主题在讲什么，25–45 字",
    "children": [
      {"title": "要点，15–28 字，要含关键公式或结论",
       "detail": "这个要点的解释或推导要点，30–60 字",
       "quote": "原文里逐字摘出的 8–20 字片段，用于在正文中定位"}
    ]}
 ]}

硬性要求：
1. 一级主题 3–8 个，按正文出现顺序排列，**优先用正文的章级标题**
   （如「振动与波动」「静电场」「恒定磁场」），去掉「第 4 章」这类编号。
   不要拿「一、简谐振动」这种小节名当一级主题 —— 那是要点的位置，
   放上来会让图层级混乱、同一章的内容碎成好几块。
2. 每个主题下 3–6 个要点，写知识本身而不是标题：
   好：「弹簧振子周期 T=2π√(m/k)，只与 m、k 有关」
   差：「简谐振动」「弹簧振子」（这是标题，没有信息）
3. quote 必须从正文里逐字复制 8–20 字（可以跳过排版符号），
   不能自己编，也不要跨段拼接 —— 它要能在原文里被搜到。
4. 公式写成普通文本：T=2π√(m/k)、Δφ=±2kπ，不要 LaTeX 记号。
5. 只做两层，不要第三层。用中文标点，不要 Markdown 记号。"""

MAX_CHARS = 12000

# 三个层级的节点宽度：根、主题、要点。要点最长，因为要放下公式
NODE_W = {0: 168, 1: 178, 2: 244}
COL_W = 262
ROW_H = 38
GAP = 12
# 节点里的字号（px）。模板用同一个值排版，布局用它估算能放几个字。
FONT = 12.5


def clean(node, depth: int = 0) -> dict | None:
    """把模型返回的节点洗成固定形状：title 必有，detail/quote 可选。"""
    if not isinstance(node, dict):
        return None
    title = str(node.get("title") or "").strip()
    if not title:
        return None
    item: dict = {"title": title[:44]}

    detail = str(node.get("detail") or "").strip()
    if detail:
        item["detail"] = detail[:130]

    # quote 只给叶子：主题层级的定位没有意义，点击主题不该跳到某一段
    quote = str(node.get("quote") or "").strip()
    if quote and depth >= 2:
        item["quote"] = quote[:60]

    if depth < 2:
        kids = []
        for c in (node.get("children") or [])[:9]:
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
        timeout=90,
        max_tokens=4000,
    )
    root = clean(data) if isinstance(data, dict) else None
    if not root:
        return None

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


def _fit(text: str, max_px: float, font: float = FONT) -> tuple[str, bool]:
    """按估算宽度截断文字，返回 (能显示的部分, 是否被截断)。

    不能按固定字数截：`T=2π√(m/k)` 里每个字符的宽度只有汉字的一半多一点，
    一律按 16 字截会白白浪费半截空间，或者在英文数字多的行上溢出框外。
    这里按「汉字 1 个字宽、其余 0.55」估算，够贴边又不会越界。
    """
    budget = max_px - 22          # 左右内边距 + 一点余量
    width = 0.0
    out: list[str] = []
    for ch in text:
        w = font if ord(ch) > 0x2E80 else font * 0.55
        if width + w > budget:
            return "".join(out), True
        out.append(ch)
        width += w
    return text, False


def _leaves(node: dict) -> int:
    kids = node.get("children") or []
    if not kids:
        return 1
    return sum(_leaves(k) for k in kids)


def layout(
    root: dict,
    pad_left: int = 14,
    pad_right: int = 18,
) -> dict:
    """把树排成横向图：根在左，一级主题一列，要点再一列。

    纵向位置按「叶子数」分配：一个节点占据它所有后代的连续区间，
    父节点居中对齐到子节点区间的中点 —— 这样连线不会互相穿插。
    子树之间的空隙由 GAP 控制，行高固定，形状因此可复现。
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    total = _leaves(root)
    height = max(1, total) * ROW_H + (max(1, total) - 1) * GAP
    pad_top = ROW_H / 2

    def walk(node: dict, depth: int, y0: float) -> float:
        """返回这个子树占据的中心 y。"""
        kids = node.get("children") or []
        own_h = _leaves(node) * ROW_H + (_leaves(node) - 1) * GAP
        if not kids:
            cy = y0 + ROW_H / 2
        else:
            cy = y0 + own_h / 2
            cursor = y0
            for k in kids:
                ky = walk(k, depth + 1, cursor)
                edges.append(
                    {
                        "x1": pad_left + depth * COL_W + NODE_W[min(depth, 2)],
                        "y1": cy,
                        "x2": pad_left + (depth + 1) * COL_W,
                        "y2": ky,
                    }
                )
                cursor += _leaves(k) * ROW_H + (_leaves(k) - 1) * GAP
        w = NODE_W[min(depth, 2)]
        label, cut = _fit(node["title"], w)
        nodes.append(
            {
                "x": pad_left + depth * COL_W,
                "y": round(cy, 1),
                "title": node["title"],
                "label": label,
                "ellipsis": cut,
                "detail": node.get("detail", ""),
                "quote": node.get("quote", ""),
                "depth": depth,
                "leaf": not kids,
                "w": w,
            }
        )
        return cy

    walk(root, 0, pad_top)

    width = max(n["x"] + n["w"] for n in nodes) + pad_right
    return {
        "nodes": nodes,
        "edges": edges,
        "width": int(width),
        "height": int(height + ROW_H),
    }
