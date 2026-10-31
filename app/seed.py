"""示例库载入。数据在 data/seed.json，格式见 run.md。"""
from __future__ import annotations

import datetime as dt
import json

from . import config, db, ingest

SEED_PATH = config.BASE_DIR / "data" / "seed.json"


def load_seed(user_id: int, force: bool = False) -> int:
    if not SEED_PATH.exists():
        return 0
    existing = db.one(
        "SELECT COUNT(*) AS n FROM documents WHERE user_id = ? AND source_kind = 'seed'",
        (user_id,),
    )
    if existing and existing["n"] > 0 and not force:
        return 0

    payload = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    now = dt.datetime.now().isoformat(timespec="seconds")
    count = 0

    for item in payload:
        doc_id = ingest.ingest_document(
            user_id,
            item["title"],
            item["body"],
            kind=item.get("kind", "course"),
            period=item.get("period"),
            source_kind="seed",
            source_url=item.get("source_url"),
        )
        ingest.add_tags(doc_id, user_id, item.get("tags", []))
        for card in item.get("cards", []):
            db.execute(
                """INSERT INTO cards
                   (user_id, document_id, what, result, ability, todo, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    doc_id,
                    card.get("what", ""),
                    card.get("result", ""),
                    card.get("ability", ""),
                    card.get("todo", ""),
                    now,
                ),
            )
        count += 1
    return count
