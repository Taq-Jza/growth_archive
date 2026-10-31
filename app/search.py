"""检索：关键词路（FTS5 + jieba）与语义路（bge-m3 + 暴力余弦），以及 RRF 融合。

为什么不用向量数据库：300 个切片规模下，把向量读进内存做全量余弦只要几毫秒，
比维护一个 ANN 索引更简单、更可控。真要上十万级只需替换 semantic_search 内部实现，
retrieve() 这个入口不用动。
"""
from __future__ import annotations

import json

from . import db, embed
from .textproc import cosine, highlight, keyword_terms, to_fts_query

# 每个用户一份向量缓存：(版本号, [(chunk_id, vector), ...])
_VECTOR_CACHE: dict[int, tuple[tuple[int, int], list[tuple[int, list[float]]]]] = {}


def invalidate_vector_cache(user_id: int | None = None) -> None:
    if user_id is None:
        _VECTOR_CACHE.clear()
    else:
        _VECTOR_CACHE.pop(user_id, None)


def keyword_search(
    user_id: int,
    query: str,
    limit: int = 20,
    offset: int = 0,
    kind: str | None = None,
    period_from: str | None = None,
    period_to: str | None = None,
) -> list[dict]:
    """关键词检索。

    先按 AND 求精确命中；为零时退化为 OR 求召回。
    原因：自然语言长问题（「我做过哪些数据分析相关的事」）如果全词 AND，
    几乎必然零命中；而短查询用 AND 精度更高。两种都试，代价可忽略。
    """
    terms = keyword_terms(query)
    if not terms:
        return []

    rows: list = []
    for mode in ("AND", "OR"):
        rows = _fts_query(
            user_id, to_fts_query(terms, mode), limit, offset, kind, period_from, period_to
        )
        if rows:
            break

    out = []
    for r in rows:
        out.append(
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "title": r["title"],
                "kind": r["kind"],
                "period": r["period"],
                "created_at": r["created_at"],
                "score": r["score"],
                "text": r["text"],
                "snippet": highlight(r["text"], terms),
            }
        )
    return out


def keyword_count(
    user_id: int,
    query: str,
    kind: str | None = None,
    period_from: str | None = None,
    period_to: str | None = None,
) -> int:
    """命中总数，用于分页。取不到返回 0。"""
    terms = keyword_terms(query)
    if not terms:
        return 0
    for mode in ("AND", "OR"):
        total = _fts_count(user_id, to_fts_query(terms, mode), kind, period_from, period_to)
        if total:
            return total
    return 0


def _filter_sql(kind, period_from, period_to, tag=None, q=None) -> tuple[str, list]:
    """类型 / 时间范围 / 标签筛选。

    标签必须按 document_tags 精确匹配，**不能退化成关键词搜索** ——
    像「课程」这种标签，正文里往往一次都没出现过，当关键词搜必然 0 命中。
    """
    sql = ""
    params: list = []
    if kind:
        sql += " AND d.kind = ?"
        params.append(kind)
    if tag:
        # 与「标签收藏夹」的分组口径保持一致：只认主标签（第一个非类型标签）。
        # 若按任意标签匹配，会出现文件夹标 3 条、点进去 5 条的情况。
        from .ingest import KINDS

        labels = tuple(KINDS.values())
        ph = ",".join("?" for _ in labels)
        sql += f""" AND (SELECT t2.name FROM document_tags dt2 JOIN tags t2 ON t2.id = dt2.tag_id
                          WHERE dt2.document_id = d.id AND t2.name NOT IN ({ph})
                          ORDER BY dt2.tag_id LIMIT 1) = ?"""
        params.extend(labels)
        params.append(tag)
    if q:
        # 分类页的「搜索本类记录」：只匹配标题与正文，返回的是记录而不是切片，
        # 这样列表里不会出现同一条记录被拆成好几行。
        sql += " AND (d.title LIKE ? OR d.body LIKE ?)"
        like = f"%{q}%"
        params.extend([like, like])
    if period_from:
        sql += " AND substr(COALESCE(d.period, d.created_at), 1, 10) >= ?"
        params.append(period_from)
    if period_to:
        sql += " AND substr(COALESCE(d.period, d.created_at), 1, 10) <= ?"
        params.append(period_to)
    return sql, params


def _fts_query(user_id: int, fts: str, limit: int, offset: int = 0, kind=None, period_from=None, period_to=None) -> list:
    if not fts:
        return []
    extra, extra_params = _filter_sql(kind, period_from, period_to)
    return db.rows(
        f"""
        SELECT chunks_fts.chunk_id AS chunk_id,
               bm25(chunks_fts)     AS score,
               c.text               AS text,
               c.document_id        AS document_id,
               d.title              AS title,
               d.kind               AS kind,
               d.period             AS period,
               d.created_at         AS created_at
        FROM chunks_fts
        JOIN chunks c    ON c.id = chunks_fts.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE chunks_fts MATCH ? AND d.user_id = ?{extra}
        ORDER BY score
        LIMIT ? OFFSET ?
        """,
        (fts, user_id, *extra_params, limit, offset),
    )


def _fts_count(user_id: int, fts: str, kind=None, period_from=None, period_to=None) -> int:
    if not fts:
        return 0
    extra, extra_params = _filter_sql(kind, period_from, period_to)
    row = db.one(
        f"""SELECT COUNT(*) AS n
            FROM chunks_fts
            JOIN chunks c    ON c.id = chunks_fts.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ? AND d.user_id = ?{extra}""",
        (fts, user_id, *extra_params),
    )
    return row["n"] if row else 0


def browse(
    user_id: int,
    kind: str | None = None,
    period_from: str | None = None,
    period_to: str | None = None,
    limit: int = 30,
    offset: int = 0,
    tag: str | None = None,
    sort: str = "time",
    q: str | None = None,
) -> list[dict]:
    """只筛选不搜索：按类型/时间/标签列出记录（对应 AC-03-4 的例外条款）。

    sort 支持四种：time（最新）/ time_asc（最早）/ kind（按分类）/ tag（按主标签）。
    分类页与概览共用同一套排序口径，避免两处行为不一致。
    """
    extra, params = _filter_sql(kind, period_from, period_to, tag, q)
    order = {
        "time_asc": "COALESCE(d.period, d.created_at) ASC, d.created_at ASC",
        "kind": "d.kind, COALESCE(d.period, d.created_at) DESC",
        "tag": "main_tag, COALESCE(d.period, d.created_at) DESC",
    }.get(sort, "COALESCE(d.period, d.created_at) DESC, d.created_at DESC")

    # 主标签必须排除类型名（课程/竞赛…）：类型不是「这段经历属于什么领域」的答案，
    # 拿它当分组依据，按标签排序就退化成了按分类排序。
    from .ingest import KINDS  # 延迟导入，避免与 ingest 形成模块级循环

    labels = tuple(KINDS.values())
    label_ph = ",".join("?" for _ in labels)

    rows = db.rows(
        f"""SELECT d.id AS document_id, d.title, d.kind, d.period, d.body, d.created_at,
                   (SELECT MIN(c.id) FROM chunks c WHERE c.document_id = d.id) AS chunk_id,
                   (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS n_chunks,
                   (SELECT COUNT(*) FROM cards c WHERE c.document_id = d.id) AS n_cards,
                   (SELECT t2.name FROM document_tags dt2 JOIN tags t2 ON t2.id = dt2.tag_id
                     WHERE dt2.document_id = d.id AND t2.name NOT IN ({label_ph})
                     ORDER BY dt2.tag_id LIMIT 1) AS main_tag
            FROM documents d
            WHERE d.user_id = ?{extra}
            ORDER BY {order}
            LIMIT ? OFFSET ?""",
        (*labels, user_id, *params, limit, offset),
    )
    out = []
    for r in rows:
        body = (r["body"] or "").replace("\n", " ").strip()
        out.append(
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "title": r["title"],
                "kind": r["kind"],
                "period": r["period"],
                "created_at": r["created_at"],
                "n_chunks": r["n_chunks"],
                "n_cards": r["n_cards"],
                "main_tag": r["main_tag"],
                "score": 0.0,
                "text": body[:200],
                "snippet": body[:140] + ("…" if len(body) > 140 else ""),
            }
        )
    return out


def browse_count(
    user_id: int,
    kind: str | None = None,
    period_from: str | None = None,
    period_to: str | None = None,
    tag: str | None = None,
    q: str | None = None,
) -> int:
    extra, params = _filter_sql(kind, period_from, period_to, tag, q)
    row = db.one(f"SELECT COUNT(*) AS n FROM documents d WHERE d.user_id = ?{extra}", (user_id, *params))
    return row["n"] if row else 0


def kind_breakdown(user_id: int) -> list[dict]:
    """每个分类下的标签分布，供饼图下钻使用。"""
    rows = db.rows(
        """SELECT d.kind AS kind, t.name AS name, COUNT(*) AS n
           FROM documents d
           JOIN document_tags dt ON dt.document_id = d.id
           JOIN tags t ON t.id = dt.tag_id
           WHERE d.user_id = ?
           GROUP BY d.kind, t.name
           ORDER BY d.kind, n DESC""",
        (user_id,),
    )
    return [dict(r) for r in rows]


def kind_folders(user_id: int) -> list[dict]:
    """按五类记录聚合成「文件夹」：每类一张卡，带条数与最近几条标题。

    概览的「按分类」原本是「分组标题 + 一长串记录」，记录一多就退化成瀑布流，
    找不到东西。改成文件夹后，先看到五类各多少条，点进去才是该类记录。
    """
    from .ingest import KINDS

    rows = db.rows(
        """SELECT d.id AS doc_id, d.title, d.kind, d.period, d.created_at
           FROM documents d
           WHERE d.user_id = ?
           ORDER BY COALESCE(d.period, d.created_at) DESC""",
        (user_id,),
    )
    buckets: dict[str, dict] = {}
    for r in rows:
        b = buckets.setdefault(
            r["kind"], {"kind": r["kind"], "label": KINDS.get(r["kind"], r["kind"]), "n": 0, "docs": []}
        )
        b["n"] += 1
        if len(b["docs"]) < 3:
            b["docs"].append({"id": r["doc_id"], "title": r["title"]})

    # 保持 KINDS 的固定顺序（课程→技能→竞赛→项目→特长），而不是按数量排序，
    # 侧边栏也是这个顺序，两处一致才不用重新找位置。
    return [buckets[k] for k in KINDS if k in buckets]


def tag_folders(user_id: int, kind: str | None = None) -> list[dict]:
    """按主标签聚合成「收藏夹」：每个标签一组，带记录数与最近几条标题。

    只用主标签（第一个非类型标签）分组，理由和能力域一致：
    一条记录可能挂 3 个标签，全用就会出现同一条记录出现在多个文件夹里。
    """
    from .ingest import KINDS

    labels = tuple(KINDS.values())
    ph = ",".join("?" for _ in labels)
    extra = " AND d.kind = ?" if kind else ""
    params: list = [*labels, user_id]
    if kind:
        params.append(kind)

    rows = db.rows(
        f"""SELECT d.id AS doc_id, d.title, d.period, d.created_at,
                   (SELECT t2.name FROM document_tags dt2 JOIN tags t2 ON t2.id = dt2.tag_id
                     WHERE dt2.document_id = d.id AND t2.name NOT IN ({ph})
                     ORDER BY dt2.tag_id LIMIT 1) AS main_tag
            FROM documents d
            WHERE d.user_id = ?{extra}
            ORDER BY COALESCE(d.period, d.created_at) DESC""",
        tuple(params),
    )

    buckets: dict[str, dict] = {}
    for r in rows:
        tag = r["main_tag"] or "未标注"
        b = buckets.setdefault(tag, {"name": tag, "n": 0, "docs": []})
        b["n"] += 1
        if len(b["docs"]) < 3:
            b["docs"].append({"id": r["doc_id"], "title": r["title"]})

    # 有标签的排在前面，未标注沉底
    return sorted(
        buckets.values(),
        key=lambda x: (x["name"] == "未标注", -x["n"], x["name"]),
    )


def card_tag_folders(user_id: int, kind: str | None = None) -> list[dict]:
    """要点卡的标签收藏夹：按卡片所属记录的主标签分组，计数是卡片数。"""
    from .ingest import KINDS

    labels = tuple(KINDS.values())
    ph = ",".join("?" for _ in labels)
    extra = " AND d.kind = ?" if kind else ""
    params: list = [*labels, user_id]
    if kind:
        params.append(kind)

    rows = db.rows(
        f"""SELECT d.id AS doc_id, d.title,
                   (SELECT COUNT(*) FROM cards c WHERE c.document_id = d.id) AS n_cards,
                   (SELECT t2.name FROM document_tags dt2 JOIN tags t2 ON t2.id = dt2.tag_id
                     WHERE dt2.document_id = d.id AND t2.name NOT IN ({ph})
                     ORDER BY dt2.tag_id LIMIT 1) AS main_tag
            FROM documents d
            WHERE d.user_id = ?{extra}
              AND EXISTS (SELECT 1 FROM cards c WHERE c.document_id = d.id)
            ORDER BY COALESCE(d.period, d.created_at) DESC""",
        tuple(params),
    )

    buckets: dict[str, dict] = {}
    for r in rows:
        tag = r["main_tag"] or "未标注"
        b = buckets.setdefault(tag, {"name": tag, "n": 0, "docs": []})
        b["n"] += r["n_cards"]
        if len(b["docs"]) < 3:
            b["docs"].append({"id": r["doc_id"], "title": r["title"]})
    return sorted(
        buckets.values(),
        key=lambda x: (x["name"] == "未标注", -x["n"], x["name"]),
    )


def capability_buckets(
    user_id: int,
    mapping: dict[str, list[str]],
    kind_labels: tuple[str, ...] = (),
) -> list[dict]:
    """按能力域聚合。

    每条记录取它的**主标签**（第一个非类型标签），再按映射表归到能力域。
    没有主标签或标签不在映射表里的记录归入「其他」——这样扇区之和恒等于记录总数，
    百分比才是真的。
    """
    tag_to_bucket: dict[str, str] = {}
    for bucket, tags in mapping.items():
        if bucket.startswith("_"):
            continue
        for t in tags:
            tag_to_bucket[t] = bucket

    ph = ",".join("?" for _ in kind_labels) if kind_labels else None
    sql = f"""SELECT d.id AS id, d.title AS title, d.period AS period,
                     d.created_at AS created_at, t.name AS tag, dt.tag_id AS tid
              FROM documents d
              JOIN document_tags dt ON dt.document_id = d.id
              JOIN tags t ON t.id = dt.tag_id
              WHERE d.user_id = ?"""
    params: list = [user_id]
    if ph:
        sql += f" AND t.name NOT IN ({ph})"
        params.extend(kind_labels)
    sql += " ORDER BY d.id, dt.tag_id"

    first_tag: dict[int, dict] = {}
    for r in db.rows(sql, tuple(params)):
        first_tag.setdefault(r["id"], dict(r))

    docs = db.rows(
        """SELECT id, title, period, created_at FROM documents WHERE user_id = ?
           ORDER BY COALESCE(period, created_at) DESC""",
        (user_id,),
    )

    buckets: dict[str, dict] = {}
    for d in docs:
        hit = first_tag.get(d["id"])
        bucket = tag_to_bucket.get(hit["tag"], "其他") if hit else "其他"
        b = buckets.setdefault(bucket, {"name": bucket, "n": 0, "tags": {}, "records": []})
        b["n"] += 1
        if hit:
            b["tags"][hit["tag"]] = b["tags"].get(hit["tag"], 0) + 1
        if len(b["records"]) < 4:
            b["records"].append({"id": d["id"], "title": d["title"], "period": d["period"]})

    out = sorted(buckets.values(), key=lambda x: (-x["n"], x["name"]))
    for b in out:
        b["tags"] = [
            {"name": k, "n": v}
            for k, v in sorted(b["tags"].items(), key=lambda kv: (-kv[1], kv[0]))
        ]
    return out


def records_by_tag(user_id: int, tag: str, limit: int = 4) -> list[dict]:
    return [
        dict(r)
        for r in db.rows(
            """SELECT d.id AS id, d.title AS title, d.period AS period
               FROM documents d
               JOIN document_tags dt ON dt.document_id = d.id
               JOIN tags t ON t.id = dt.tag_id
               WHERE d.user_id = ? AND t.name = ?
               ORDER BY COALESCE(d.period, d.created_at) DESC
               LIMIT ?""",
            (user_id, tag, limit),
        )
    ]


def most_recent_documents(user_id: int, limit: int = 4, only_untagged: bool = False) -> list[dict]:
    extra = (
        " AND d.id NOT IN (SELECT document_id FROM document_tags)" if only_untagged else ""
    )
    return [
        dict(r)
        for r in db.rows(
            f"""SELECT d.id AS id, d.title AS title, d.period AS period
                FROM documents d
                WHERE d.user_id = ?{extra}
                ORDER BY COALESCE(d.period, d.created_at) DESC
                LIMIT ?""",
            (user_id, limit),
        )
    ]


def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60, limit: int = 12) -> list[dict]:
    """Reciprocal Rank Fusion：把多路排序结果合成一个。"""
    merged: dict[int, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            cid = item["chunk_id"]
            entry = merged.setdefault(cid, {**item, "rrf": 0.0})
            entry["rrf"] += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)[:limit]


def _load_vectors(user_id: int) -> list[tuple[int, list[float]]]:
    """带版本校验的向量缓存。切片数或最大 ID 变化即失效。"""
    meta = db.one(
        """SELECT COUNT(*) AS n, COALESCE(MAX(c.id), 0) AS m
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE d.user_id = ? AND c.embedding IS NOT NULL""",
        (user_id,),
    )
    version = (meta["n"], meta["m"])
    cached = _VECTOR_CACHE.get(user_id)
    if cached and cached[0] == version:
        return cached[1]

    rows = db.rows(
        """SELECT c.id AS id, c.embedding AS emb
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE d.user_id = ? AND c.embedding IS NOT NULL""",
        (user_id,),
    )
    vectors: list[tuple[int, list[float]]] = []
    for row in rows:
        try:
            vectors.append((row["id"], json.loads(row["emb"])))
        except (TypeError, json.JSONDecodeError):
            continue
    _VECTOR_CACHE[user_id] = (version, vectors)
    return vectors


def semantic_search(user_id: int, query: str, limit: int = 20, kind: str | None = None) -> list[dict]:
    """语义检索。任何失败都返回空列表，由调用方退回关键词路。"""
    if not embed.available():
        return []
    vectors = _load_vectors(user_id)
    if not vectors:
        return []

    try:
        query_vec = embed.embed_query(query)
    except Exception:  # noqa: BLE001
        return []

    scored = []
    for chunk_id, vec in vectors:
        score = cosine(query_vec, vec)
        if score > 0:
            scored.append((score, chunk_id))
    scored.sort(reverse=True)
    top = scored[: limit * 3 if kind else limit]
    if not top:
        return []

    ids = [cid for _, cid in top]
    placeholders = ",".join("?" for _ in ids)
    extra, extra_params = _filter_sql(kind, None, None)
    rows = db.rows(
        f"""SELECT c.id AS chunk_id, c.text AS text, c.document_id AS document_id,
                   d.title AS title, d.kind AS kind, d.period AS period, d.created_at AS created_at
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders}){extra}""",
        (*ids, *extra_params),
    )
    by_id = {r["chunk_id"]: r for r in rows}
    terms = keyword_terms(query)

    out: list[dict] = []
    for score, cid in top:
        row = by_id.get(cid)
        if not row:
            continue
        out.append(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "kind": row["kind"],
                "period": row["period"],
                "created_at": row["created_at"],
                "score": -score,
                "text": row["text"],
                "snippet": highlight(row["text"], terms),
            }
        )
        if len(out) >= limit:
            break
    return out
