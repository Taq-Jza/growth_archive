"""极简会话认证：PBKDF2 存密码 + itsdangerous 签名 Cookie。

做四件事：自建账号、注册体验账号、登录、登出。
不做：邮箱验证、找回密码、第三方登录、多设备管理 —— 这些都要额外的外部服务，
在单机部署里换不来可见的差别。忘了口令就只能重建账号，页面上写清楚。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import re
import sqlite3

from fastapi import Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from . import config, db

COOKIE_NAME = "sdd_session"
MAX_AGE = 60 * 60 * 24 * 30
_serializer = URLSafeTimedSerializer(config.SECRET_KEY, salt="session")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120_000)
    return hmac.compare_digest(digest.hex(), digest_hex)


# 只做「像不像邮箱」的粗检：要真正验证得发信，那是另一个量级的事。
# 这里挡的是明显的手滑（漏 @、漏域名），不是挡人。
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def register_user(email: str, password: str) -> tuple[dict | None, str]:
    """自建账号。返回 (用户, 错误说明)，二选一。

    口令只存 PBKDF2 派生值，不存原文 —— 这是唯一不能省的地方。
    """
    email = (email or "").strip().lower()
    password = password or ""

    if not _EMAIL_RE.match(email):
        return None, "邮箱格式不对，检查是不是漏了 @ 或域名"
    if len(email) > 120:
        return None, "邮箱太长了"
    if len(password) < 6:
        return None, "口令至少 6 位"
    if len(password) > 128:
        return None, "口令太长了，128 位以内"
    if db.one("SELECT id FROM users WHERE email = ?", (email,)):
        return None, "这个邮箱已经注册过，直接登录就行"

    try:
        db.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, hash_password(password), dt.datetime.now().isoformat(timespec="seconds")),
        )
    except sqlite3.IntegrityError:
        # 两个人同时用同一个邮箱注册时会走到这里（UNIQUE 兜底）
        return None, "这个邮箱已经注册过，直接登录就行"

    row = db.one("SELECT id, email FROM users WHERE email = ?", (email,))
    return (dict(row) if row else None), ""


def ensure_demo_user() -> None:
    """首次启动自动建体验账号，新用户点一下就能进，不需要注册。"""
    existing = db.one("SELECT id FROM users WHERE email = ?", (config.DEMO_EMAIL,))
    if existing:
        return
    db.execute(
        "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
        (config.DEMO_EMAIL, hash_password(config.DEMO_PASSWORD), dt.datetime.now().isoformat()),
    )


def login_user(email: str, password: str) -> dict | None:
    row = db.one("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return dict(row)


def make_cookie(user_id: int) -> str:
    return _serializer.dumps({"uid": user_id})


def current_user(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        data = _serializer.loads(token, max_age=MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    row = db.one("SELECT id, email FROM users WHERE id = ?", (data.get("uid"),))
    return dict(row) if row else None
