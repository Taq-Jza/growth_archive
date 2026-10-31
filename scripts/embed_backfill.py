"""给已有切片补向量。换 embedding 模型、或某次入库时接口失败，都跑这个。

用法：
    python scripts/embed_backfill.py            # 补齐所有缺失的
    python scripts/embed_backfill.py --rebuild  # 全部重算（换模型后必须用这个）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, embed  # noqa: E402


def main() -> int:
    rebuild = "--rebuild" in sys.argv

    if not embed.available():
        print("[X] 未配置 LLM_API_KEY，无法生成向量。")
        return 1

    if rebuild:
        with db.get_conn() as conn:
            conn.execute("UPDATE chunks SET embedding = NULL")
        print("已清空全部向量，准备重算。")

    rows = db.rows(
        """SELECT c.id AS id, c.text AS text
           FROM chunks c WHERE c.embedding IS NULL"""
    )
    if not rows:
        print("没有需要处理的切片。")
        return 0

    print(f"待处理 {len(rows)} 个切片，模型 {config.EMBED_MODEL} …")
    t0 = time.time()
    done = 0
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        try:
            vectors = embed.embed_texts([r["text"] for r in batch])
        except Exception as exc:  # noqa: BLE001
            print(f"[!] 第 {start} 批失败：{exc}")
            continue
        with db.get_conn() as conn:
            for row, vec in zip(batch, vectors):
                conn.execute(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    (json.dumps(vec), row["id"]),
                )
        done += len(batch)
        print(f"  已处理 {done}/{len(rows)}")

    print(f"完成：{done} 个切片，耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
