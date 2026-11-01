"""SQLite 数据层。

【为什么不用 PostgreSQL + pgvector】
当前规模：单用户、数十条记录、千级切片。
- SQLite 零运维，ECS 上不用额外装数据库，省掉约 5 小时环境调试；
- 200 条向量的余弦相似度在 Python 里暴力算是毫秒级；
- FTS5 + jieba 预分词足以支撑中文检索。
在 5 天工期里，「能跑起来」比「架构正确」值钱。迁移路径见 run.md。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterable

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL DEFAULT 'course',
    title      TEXT NOT NULL,
    body       TEXT NOT NULL,
    period     TEXT,
    source_kind TEXT NOT NULL DEFAULT 'paste',
    source_url TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_user ON documents(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_doc_kind ON documents(user_id, kind);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    seg_text    TEXT NOT NULL,
    embedding   TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    seg_text,
    tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS tags (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name    TEXT NOT NULL,
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS document_tags (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, tag_id)
);

CREATE TABLE IF NOT EXISTS cards (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    document_id    INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    what           TEXT NOT NULL,
    result         TEXT NOT NULL DEFAULT '',
    ability        TEXT NOT NULL DEFAULT '',
    todo           TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_card_user ON cards(user_id, created_at DESC);

-- 记录的原图 / 重绘图：正文是检索用的文本，图片只是阅读时的补充
CREATE TABLE IF NOT EXISTS attachments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    role        TEXT NOT NULL DEFAULT 'original',   -- original | redrawn | inline
    filename    TEXT NOT NULL,
    mime        TEXT,
    seq         INTEGER NOT NULL DEFAULT 0,         -- inline 图在正文里的序号
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_att_doc ON attachments(document_id);

-- 知识图谱的中间层：每条记录的章节与关键概念（由模型从正文抽取）
CREATE TABLE IF NOT EXISTS doc_kg (
    document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    chapter     TEXT NOT NULL DEFAULT '',
    concepts    TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL
);

-- 思维导图：一份资料自身的知识结构（与 doc_kg 的单章节不同，这里是一棵树）
CREATE TABLE IF NOT EXISTS doc_map (
    document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    data        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

-- 面试预演的上一次结果。body_hash 用来判断记录改过没有：
-- 正文一改，题目就不再对应，缓存自动作废。
CREATE TABLE IF NOT EXISTS interview_cache (
    document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    user_id     INTEGER NOT NULL,
    body_hash   TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """就地升级旧数据。

    1) period 早期只存到月（2026-03），日期筛选精确到日后统一补成 2026-03-01。
       用 length(period)=7 做条件，重复执行不会二次追加。
    """
    conn.execute(
        "UPDATE documents SET period = period || '-01' "
        "WHERE period IS NOT NULL AND length(period) = 7"
    )

    # 2) 类型名曾被当作标签存进来（如「课程」「竞赛」），会污染标签云与能力域统计。
    #    这里清掉关联，标签本体保留（可能被别的记录真正用作主题词的情况不存在，一并删）。
    labels = ("课程", "技能", "竞赛", "项目", "特长")
    ph = ",".join("?" for _ in labels)
    conn.execute(
        f"DELETE FROM document_tags WHERE tag_id IN (SELECT id FROM tags WHERE name IN ({ph}))",
        labels,
    )
    conn.execute(f"DELETE FROM tags WHERE name IN ({ph})", labels)

    # 3) attachments.seq：正文里的内嵌图片按序号占位，渲染时按 seq 取回。
    #    CREATE TABLE IF NOT EXISTS 不会给已有表补列，只能在这里 ALTER。
    cols = {r[1] for r in conn.execute("PRAGMA table_info(attachments)")}
    if "seq" not in cols:
        conn.execute("ALTER TABLE attachments ADD COLUMN seq INTEGER NOT NULL DEFAULT 0")


def rows(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(sql, tuple(params)).fetchall()


def one(sql: str, params: Iterable[Any] = ()):
    with get_conn() as conn:
        return conn.execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    with get_conn() as conn:
        cur = conn.execute(sql, tuple(params))
        return cur.lastrowid or 0
