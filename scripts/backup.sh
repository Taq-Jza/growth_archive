#!/usr/bin/env bash
# 备份：数据库整库快照 + 示例数据打包，保留最近 7 份
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$APP_DIR/backups}"
KEEP="${KEEP:-7}"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$BACKUP_DIR"

if [ -f "$APP_DIR/data/app.db" ]; then
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$APP_DIR/data/app.db" ".backup '$BACKUP_DIR/app-$STAMP.db'"
  else
    cp "$APP_DIR/data/app.db" "$BACKUP_DIR/app-$STAMP.db"
  fi
  echo "数据库已备份 → app-$STAMP.db"
else
  echo "未找到 data/app.db，跳过数据库备份"
fi

tar -czf "$BACKUP_DIR/assets-$STAMP.tar.gz" \
  -C "$APP_DIR" data/seed.json data/jd_samples.json .env.example 2>/dev/null || true
echo "静态数据已备份 → assets-$STAMP.tar.gz"

ls -1t "$BACKUP_DIR"/app-*.db 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
ls -1t "$BACKUP_DIR"/assets-*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
echo "已保留最近 ${KEEP} 份"
