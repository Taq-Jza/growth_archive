"""模型接入层。

只暴露 chat / chat_stream / available 三个能力，上层不关心用的是哪家。
默认走硅基流动（OpenAI 兼容）：chat 用 Qwen，embedding 用 bge-m3。

设计要点（对应《设计方案》D-04）：
- 未配置 key 时不抛异常、不阻塞启动，而是让调用方走降级路径；
- 所有超时都由调用方通过参数控制，不在这一层做重试；
- 这一层是唯一知道「外部服务存在」的地方，换供应商只改这里。
"""
from __future__ import annotations

import json
import urllib.request
from typing import Iterator

from openai import OpenAI

from . import config


class LLMUnavailable(RuntimeError):
    """模型不可用（无 key / 网络失败 / 超时）。调用方应降级而不是报错给用户。"""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY or "not-configured",
            timeout=config.LLM_TIMEOUT,
            max_retries=0,
        )
    return _client


def available() -> bool:
    return bool(config.LLM_API_KEY)


def chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: float | None = None,
    model: str | None = None,
) -> str:
    if not available():
        raise LLMUnavailable("未配置 LLM_API_KEY")
    try:
        resp = _get_client().chat.completions.create(
            model=model or config.CHAT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout or config.LLM_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 网络/鉴权/限流统一降级
        raise LLMUnavailable(str(exc)[:200]) from exc
    return (resp.choices[0].message.content or "").strip()


def chat_vision(
    image_b64: str,
    prompt: str,
    mime: str = "image/jpeg",
    model: str | None = None,
    max_tokens: int = 4000,
    timeout: float | None = None,
) -> str:
    """图片理解 / OCR。走 OpenAI 兼容的多模态消息格式。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ],
        }
    ]
    return chat(
        messages,
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=timeout or max(config.LLM_TIMEOUT, 90),
        model=model or config.VISION_MODEL,
    )


def edit_image(image_b64: str, prompt: str, mime: str = "image/jpeg") -> bytes:
    """图像重绘：调用图生图模型，返回图片二进制。

    走的是 OpenAI 兼容的 /images/generations（SiliconFlow 的图生图也挂在这里，
    通过 image 字段传原图）。返回的是临时 URL，必须当场下载 —— 一小时后失效。
    """
    if not available():
        raise LLMUnavailable("未配置 LLM_API_KEY")
    payload = {
        "model": config.IMAGE_MODEL,
        "prompt": prompt,
        "image": f"data:{mime};base64,{image_b64}",
        "image_size": config.IMAGE_SIZE,
        "batch_size": 1,
    }
    req = urllib.request.Request(
        config.LLM_BASE_URL.rstrip("/") + "/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        url = data["images"][0]["url"]
        with urllib.request.urlopen(url, timeout=90) as resp:
            return resp.read()
    except Exception as exc:  # noqa: BLE001 网络/限流/内容审核统一降级
        raise LLMUnavailable(str(exc)[:200]) from exc


def chat_stream(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: float | None = None,
) -> Iterator[str]:
    if not available():
        raise LLMUnavailable("未配置 LLM_API_KEY")
    try:
        stream = _get_client().chat.completions.create(
            model=config.CHAT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            timeout=timeout or config.LLM_TIMEOUT,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
    except LLMUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(str(exc)[:200]) from exc


def chat_json(
    messages: list[dict], timeout: float | None = None, max_tokens: int = 1200
) -> dict | None:
    """要求模型返回 JSON，解析失败返回 None（由调用方决定重试或降级）。

    max_tokens 要由调用方按输出规模给：思维导图那种多节点嵌套的 JSON 有四千多字，
    沿用 chat() 的默认值会在半截被截断，解析出来是 None ——
    表现成「生成失败」，但日志里没有任何异常，最难查。
    """
    raw = chat(messages, temperature=0.0, timeout=timeout, max_tokens=max_tokens)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        # 也接受数组
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end <= start:
            return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
