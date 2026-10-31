"""文本处理：中文分词、切块、高亮。

中文检索的核心技巧：不用 PG 的 zhparser（编译麻烦），
而是在 Python 侧用 jieba 先切好词，按空格拼成串存进 FTS5。
unicode61 分词器会把中文串当成一个个 token，检索效果对 25 篇规模完全够。
"""
from __future__ import annotations

import re

import jieba

from . import config

# jieba 首次分词会把词典缓存写到系统临时目录（Windows 上就是 C 盘）。
# 这个项目的约定是运行产物都留在项目目录内，所以把缓存指到 data/ 下。
_CACHE_DIR = config.DATA_DIR / ".jieba"
try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(_CACHE_DIR)
except OSError:
    pass  # 建不出来就让它用默认目录，不能因为缓存位置把分词搞挂

_CJK = re.compile(r"[\u4e00-\u9fff]")


def segment(text: str) -> str:
    """切词并用空格拼接，供 FTS5 索引。"""
    if not text:
        return ""
    tokens = [t.strip() for t in jieba.cut_for_search(text) if t.strip()]
    return " ".join(tokens)


def keyword_terms(query: str) -> list[str]:
    """查询词切分。

    两个必要处理：
    1. 去重 —— jieba 的搜索模式会同时给出「可视」「可视化」，重复计数会让命中门槛虚高；
    2. 去子串 —— 「可视」是「可视化」的一部分，保留短词只会污染检索。
    """
    raw: list[str] = []
    for t in jieba.cut_for_search(query):
        t = t.strip()
        if not t or t in {"的", "了", "是", "在", "和", "我", "有", "就", "与", "及"}:
            continue
        if len(t) == 1 and not _CJK.match(t):
            continue
        if t not in raw:
            raw.append(t)

    # 只去重，不做子串裁剪：裁掉「数据」这类短词会明显伤害召回
    return raw


def to_fts_query(terms: list[str], mode: str = "AND") -> str:
    """FTS5 查询串。mode=AND 求精确，mode=OR 求召回。"""
    safe = [f'"{t.replace(chr(34), "")}"' for t in terms if t]
    return f" {mode} ".join(safe)


def chunk_text(text: str) -> list[str]:
    """按标点/段落边界切块，块之间保留重叠。

    关键约束：**切出的每一块必须是原文的精确子串**（不重排、不补空白）。
    这样「引用可核对」才是天然成立的 —— 只要 chunk 是 body 的子串，
    引用片段就一定能逐字比对通过（AC-05-4），跳转高亮也一定能定位（AC-05-2）。
    """
    text = (text or "").strip()
    if not text:
        return []

    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    boundaries = ("\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";")
    chunks: list[str] = []
    start, n = 0, len(text)

    while start < n:
        end = min(start + size, n)
        if end < n:
            segment = text[start:end]
            cut = -1
            for sep in boundaries:
                idx = segment.rfind(sep)
                if idx > cut:
                    cut = idx
            # 边界太靠前会导致块过短，宁可硬切也不产出碎片
            if cut >= int(size * 0.6):
                end = start + cut + 1
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def highlight(text: str, terms: list[str], radius: int = 90) -> str:
    """截取命中片段并把关键词包成 <mark>。返回已转义的安全 HTML。"""
    import html as _html

    if not terms:
        return _html.escape(text[: radius * 2])

    lower = text.lower()
    pos = -1
    for t in terms:
        pos = lower.find(t.lower())
        if pos >= 0:
            break
    if pos < 0:
        return _html.escape(text[: radius * 2]) + ("…" if len(text) > radius * 2 else "")

    start = max(0, pos - radius // 2)
    end = min(len(text), pos + radius)
    snippet = text[start:end]
    escaped = _html.escape(snippet)

    for t in sorted({t for t in terms if t}, key=len, reverse=True):
        if not t:
            continue
        escaped = re.sub(
            f"({re.escape(_html.escape(t))})",
            r"<mark>\1</mark>",
            escaped,
            flags=re.IGNORECASE,
        )
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + escaped + suffix


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
