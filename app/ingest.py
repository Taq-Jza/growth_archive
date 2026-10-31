"""入库：切块、分词、写索引、生成向量，以及记录的更新与删除。

索引与向量都是派生数据，所以「改一条记录」的正确做法是：
删掉它的切片和索引 → 重写正文 → 重新切块建索引 → 重新生成向量 → 清掉过期的要点卡。
"""
from __future__ import annotations

import datetime as dt
import json

from . import db, embed
from .textproc import chunk_text, segment

KINDS = {
    "course": "课程",
    "skill": "技能",
    "award": "竞赛",
    "project": "项目",
    "trait": "特长",
}


def _index_document(doc_id: int, title: str, body: str, now: str) -> None:
    """为文档重建切片与全文索引（调用前须已清空旧数据）。"""
    pieces = chunk_text(body)
    with db.get_conn() as conn:
        for seq, piece in enumerate(pieces):
            # 标题也要能被搜到：只并进第一个切片，避免每个切片都带着标题词造成噪声
            seg = segment(f"{title}\n{piece}" if seq == 0 else piece)
            cur = conn.execute(
                """INSERT INTO chunks (document_id, seq, text, seg_text, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, seq, piece, seg, now),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, seg_text) VALUES (?, ?)",
                (cur.lastrowid, seg),
            )


def _clear_derived(doc_id: int) -> None:
    """清掉一条记录的派生数据：索引、切片、要点卡。"""
    with db.get_conn() as conn:
        for row in conn.execute(
            "SELECT id FROM chunks WHERE document_id = ?", (doc_id,)
        ).fetchall():
            conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (row["id"],))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM cards WHERE document_id = ?", (doc_id,))


def _norm_period(value: str | None) -> str | None:
    """日期统一成 YYYY-MM-DD。

    早期只记到月（2026-03），补齐为当月 1 日；空值返回 None，
    由查询侧回退到创建时间。所有入库路径都过这里，避免半新半旧的数据混在一起。
    """
    if not value:
        return None
    v = str(value).strip()
    if not v:
        return None
    if len(v) == 7:
        return v + "-01"
    return v[:10]


def ingest_document(
    user_id: int,
    title: str,
    body: str,
    kind: str = "course",
    period: str | None = None,
    source_kind: str = "paste",
    source_url: str | None = None,
) -> int:
    if kind not in KINDS:
        raise ValueError(f"非法类型：{kind}")
    period = _norm_period(period)
    now = dt.datetime.now().isoformat(timespec="seconds")
    doc_id = db.execute(
        """INSERT INTO documents
           (user_id, kind, title, body, period, source_kind, source_url, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, kind, title.strip()[:200], body, period, source_kind, source_url, now),
    )
    _index_document(doc_id, title.strip()[:200], body, now)
    embed_document(doc_id, user_id)
    return doc_id


def update_document(
    user_id: int,
    doc_id: int,
    title: str,
    body: str,
    kind: str,
    period: str | None,
) -> bool:
    """改一条记录：重建切片与索引、重新生成向量、清掉过期要点卡。"""
    if kind not in KINDS:
        raise ValueError(f"非法类型：{kind}")
    doc = db.one("SELECT id FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    if not doc:
        return False

    period = _norm_period(period)
    now = dt.datetime.now().isoformat(timespec="seconds")
    _clear_derived(doc_id)
    db.execute(
        """UPDATE documents
           SET title = ?, body = ?, kind = ?, period = ?, created_at = created_at
           WHERE id = ? AND user_id = ?""",
        (title.strip()[:200], body, kind, period, doc_id, user_id),
    )
    _index_document(doc_id, title.strip()[:200], body, now)
    embed_document(doc_id, user_id)
    return True


def delete_document(user_id: int, doc_id: int) -> bool:
    doc = db.one("SELECT id FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    if not doc:
        return False

    _clear_derived(doc_id)
    with db.get_conn() as conn:
        conn.execute("DELETE FROM document_tags WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))

    from .search import invalidate_vector_cache

    invalidate_vector_cache(user_id)
    return True


def embed_document(doc_id: int, user_id: int | None = None) -> int:
    """给一篇文档的所有切片生成向量。

    失败不影响入库：记录已经在库里、关键词也能搜到，只是这一篇暂时不参与语义检索。
    之后可以跑 scripts/embed_backfill.py 补上。
    """
    if not embed.available():
        return 0
    rows = db.rows("SELECT id, text, embedding FROM chunks WHERE document_id = ?", (doc_id,))
    pending = [r for r in rows if not r["embedding"]]
    if not pending:
        return 0

    try:
        vectors = embed.embed_texts([r["text"] for r in pending])
    except Exception:  # noqa: BLE001 向量失败不阻断入库
        return 0

    with db.get_conn() as conn:
        for row, vec in zip(pending, vectors):
            conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (json.dumps(vec), row["id"]),
            )

    if user_id is not None:
        from .search import invalidate_vector_cache

        invalidate_vector_cache(user_id)
    return len(pending)


def add_tags(document_id: int, user_id: int, names: list[str]) -> None:
    """写入标签。

    类型名（课程/技能/竞赛/项目/特长）不进标签表：它们是分类维度，
    一旦混进来，标签云、按标签排序、能力域统计都会被污染。
    """
    blocked = set(KINDS.values())
    with db.get_conn() as conn:
        for name in names:
            name = name.strip()
            if not name or name in blocked:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO tags (user_id, name) VALUES (?, ?)", (user_id, name)
            )
            row = conn.execute(
                "SELECT id FROM tags WHERE user_id = ? AND name = ?", (user_id, name)
            ).fetchone()
            conn.execute(
                "INSERT OR IGNORE INTO document_tags (document_id, tag_id) VALUES (?, ?)",
                (document_id, row["id"]),
            )


def set_tags(document_id: int, user_id: int, names: list[str]) -> None:
    """覆盖式设置标签（编辑记录时用）。"""
    with db.get_conn() as conn:
        conn.execute("DELETE FROM document_tags WHERE document_id = ?", (document_id,))
    add_tags(document_id, user_id, names)


def doc_tags(document_id: int) -> list[str]:
    return [r["name"] for r in db.rows(
        """SELECT t.name FROM tags t JOIN document_tags dt ON dt.tag_id = t.id
           WHERE dt.document_id = ?""",
        (document_id,),
    )]


def rebuild_fts() -> None:
    """索引重建：清空 FTS 后按 chunks 重灌。改分词策略后调用。"""
    with db.get_conn() as conn:
        conn.execute("DELETE FROM chunks_fts")
        rows = conn.execute(
            """SELECT c.id AS cid, c.text AS text, c.seq AS seq, d.title AS title
               FROM chunks c JOIN documents d ON d.id = c.document_id"""
        ).fetchall()
        for r in rows:
            seg = segment(f"{r['title']}\n{r['text']}" if r["seq"] == 0 else r["text"])
            conn.execute("UPDATE chunks SET seg_text = ? WHERE id = ?", (seg, r["cid"]))
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, seg_text) VALUES (?, ?)",
                (r["cid"], seg),
            )
