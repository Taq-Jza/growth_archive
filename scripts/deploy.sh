#!/usr/bin/env bash
# 一键部署：装依赖 → 检查配置 → 重启服务 → 自检
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

PORT="${PORT:-8000}"
PY="${PYTHON:-python3}"

echo "[1/5] 准备虚拟环境"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

echo "[2/5] 检查配置"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  已生成 .env，请填写 LLM_API_KEY 后重新执行本脚本"
  exit 1
fi
if ! grep -qE '^LLM_API_KEY=.+' .env; then
  echo "  .env 里的 LLM_API_KEY 为空。检索与浏览可用，但问答与匹配会降级。"
fi

echo "[3/5] 停止旧进程"
mkdir -p logs
if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet growth-archive 2>/dev/null; then
  sudo systemctl restart growth-archive
  echo "  已通过 systemd 重启"
  MODE=systemd
else
  pkill -f "uvicorn app.main:app" 2>/dev/null || true
  sleep 1
  MODE=nohup
fi

echo "[4/5] 启动服务"
if [ "$MODE" = nohup ]; then
  nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1 \
    >> logs/app.log 2>&1 &
  sleep 2
fi

echo "[5/5] 自检"
if curl -fsS --max-time 8 "http://127.0.0.1:${PORT}/healthz" >/dev/null; then
  IP="$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || echo '<服务器IP>')"
  echo "  健康检查通过 → http://${IP}:${PORT}"
  echo "  查看日志：tail -f ${APP_DIR}/logs/app.log"
else
  echo "  健康检查失败，请查看 logs/app.log"
  exit 1
fi
