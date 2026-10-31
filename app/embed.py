"""向量化。硅基流动的 bge-m3，OpenAI 兼容的 embeddings 接口。

设计原则：
- 失败一律降级，绝不让「生成不了向量」拖垮「写不进记录」；
- 批量提交，一次最多 32 条，减少往返；
- 与 llm.py 共用同一套 base_url / api_key，不额外引入配置项。
"""
from __future__ import annotations

from openai import OpenAI

from . import config, llm

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
    return llm.available()


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    if not available():
        raise llm.LLMUnavailable("未配置模型密钥")

    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = [t[:2000] for t in texts[start : start + batch_size]]
            resp = _get_client().embeddings.create(
                model=config.EMBED_MODEL,
                input=batch,
                timeout=config.LLM_TIMEOUT,
            )
            vectors.extend([item.embedding for item in resp.data])
    except Exception as exc:  # noqa: BLE001
        raise llm.LLMUnavailable(f"embedding 失败：{str(exc)[:120]}") from exc
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
