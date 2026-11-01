"""联网检索：给面试预演补充「最近真的在问什么」。

为什么需要这一层：模型的知识有截止时间，而面试题每年在变。只靠模型，
问出来的还是三年前的「请介绍一下你自己」；接一路搜索，就能把近期真题
摘几条出来作为出题参考。

三家都用 HTTP + JSON，不引 SDK（多一个依赖只为拼一个请求体不划算）：
- `bocha`  国内，中文资料全，返回 webPages.value
- `serper` Google，返回 organic
- `tavily` 面向 RAG，直接给 content 摘要

没配 key 就返回空列表，调用方按「未联网」降级 —— 这是增强项，
不该因为少一个配置就让整个面试预演用不了。任何网络错误也按空处理。
"""
from __future__ import annotations

import json
import urllib.request

from . import config

MAX_SNIPPET = 220


def available() -> bool:
    return bool(config.SEARCH_API_KEY and config.SEARCH_PROVIDER)


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def _trim(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text[:MAX_SNIPPET]


def _bocha(query: str, count: int, timeout: float) -> list[dict]:
    data = _post(
        "https://api.bochaai.com/v1/web-search",
        {"query": query, "count": count, "summary": True},
        {"Authorization": f"Bearer {config.SEARCH_API_KEY}"},
        timeout,
    )
    pages = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
    return [
        {
            "title": _trim(p.get("name")),
            "snippet": _trim(p.get("summary") or p.get("snippet")),
            "url": p.get("url") or "",
        }
        for p in pages[:count]
        if p.get("snippet") or p.get("summary")
    ]


def _serper(query: str, count: int, timeout: float) -> list[dict]:
    data = _post(
        "https://google.serper.dev/search",
        {"q": query, "num": count},
        {"X-API-KEY": config.SEARCH_API_KEY},
        timeout,
    )
    return [
        {"title": _trim(it.get("title")), "snippet": _trim(it.get("snippet")), "url": it.get("link") or ""}
        for it in (data.get("organic") or [])[:count]
        if it.get("snippet")
    ]


def _tavily(query: str, count: int, timeout: float) -> list[dict]:
    data = _post(
        "https://api.tavily.com/search",
        {"api_key": config.SEARCH_API_KEY, "query": query, "max_results": count},
        {},
        timeout,
    )
    return [
        {"title": _trim(it.get("title")), "snippet": _trim(it.get("content")), "url": it.get("url") or ""}
        for it in (data.get("results") or [])[:count]
        if it.get("content")
    ]


_PROVIDERS = {"bocha": _bocha, "serper": _serper, "tavily": _tavily}


def search(query: str, count: int = 6) -> list[dict]:
    """返回 [{title, snippet, url}]。未配置 / 出错 / 空结果都返回 []。"""
    fn = _PROVIDERS.get(config.SEARCH_PROVIDER)
    if not fn or not config.SEARCH_API_KEY or not query.strip():
        return []
    try:
        return fn(query.strip(), max(1, min(count, 10)), config.SEARCH_TIMEOUT)
    except Exception:  # noqa: BLE001 搜索失败不影响主流程
        return []
