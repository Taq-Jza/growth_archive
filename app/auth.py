"""极简会话认证：PBKDF2 存密码 + itsdangerous 签名 Cookie。

当前只做三件事：注册体验账号、登录、登出。
不做：邮箱验证、找回密码、第三方登录、多设备管理（这些都要时间，且演示看不出差别）。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os

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
