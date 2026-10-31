"""多格式资料解析：文本 / Markdown / Word / PDF / 图片。

统一入口 `extract(filename, data)` 返回 (正文, 说明)。
图片走多模态模型转写，其余本地解析；任何失败都返回可读的原因，不抛异常给页面。
"""
from __future__ import annotations

import base64
import io
import re

from . import config, llm

MAX_IMAGE_EDGE = 1600
MAX_ATTACH_BYTES = 15 * 1024 * 1024

TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".log"}
DOCX_EXT = {".docx"}
PDF_EXT = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic"}

VISION_PROMPT = """这是一张学习资料的图片，可能是课件截图、教材页面、黑板照片或手写笔记。
请把它整理成**可以直接归档的中文笔记**，不要逐字转写、不要保留排版符号。

硬性要求：
1. 用自然语言分段叙述，每段围绕一个知识点，段与段之间空一行。
   不要用小标题、项目符号、编号列表、表格、代码块。
2. 公式和符号用普通文字表述（如「y = wx + b」「准确率 91%」），不要用 LaTeX 的 $ 记号。
3. 图中有表格或多列排版时，用句子把条目之间的关系讲清楚，不要还原成表格。
4. 原文出现的专有名词、数字、单位、结论必须原样保留，不得改写；不得补充图中没有的内容。
5. 只有确实辨认不出的具体词句才写「（字迹不清）」；整张图清晰时不要出现这个标注。
6. 只输出整理后的正文，不要「以下是」「这张图展示了」这类开场白。"""


def _normalize_ocr(text: str) -> str:
    """把模型可能残留的排版标记洗成自然语言。

    即使提示词要求纯文本，模型仍常带上 **加粗**、- 列表、| 表格 |、$公式$。
    这些符号一旦入库，会污染中文分词（检索命中率下降）和列表摘要的可读性，
    所以在入口处统一清掉，而不是指望每个下游模块自己兼容。
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # 表格行：| 项目 | 数值 | → 「项目：数值」
    def _row(m: "re.Match[str]") -> str:
        cells = [c.strip() for c in m.group(1).split("|")]
        cells = [c for c in cells if c and not set(c) <= set("-: ")]
        if not cells:
            return ""
        if len(cells) == 2:
            return f"{cells[0]}：{cells[1]}"
        return "，".join(cells)

    t = re.sub(r"^\s*\|(.+)\|\s*$", _row, t, flags=re.M)

    t = re.sub(r"^\s*[-=*_]{3,}\s*$", "", t, flags=re.M)      # 分隔线
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)            # 标题井号
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)             # 无序列表
    t = re.sub(r"^\s*\d+[.、)]\s+", "", t, flags=re.M)        # 有序列表
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)                  # 加粗
    t = re.sub(r"__(.+?)__", r"\1", t)                         # 加粗（下划线式）
    t = t.replace("`", "")                                      # 行内代码
    t = re.sub(r"\$\$(.+?)\$\$", r"\1", t, flags=re.S)      # 块级公式
    t = re.sub(r"\$(.+?)\$", r"\1", t)                        # 行内公式
    t = re.sub(r"\n{3,}", "\n\n", t)                          # 多余空行
    return t.strip()


class ExtractError(Exception):
    """解析失败，message 直接展示给用户。"""


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _from_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ExtractError("服务器缺少 python-docx，无法解析 Word 文档。") from exc

    doc = Document(io.BytesIO(data))
    parts: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                parts.append("| " + " | ".join(cells) + " |")
    return "\n\n".join(parts)


def _from_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractError("服务器缺少 pypdf，无法解析 PDF。") from exc

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ExtractError("这份 PDF 有密码保护，请先解除加密再上传。")

    pages: list[str] = []
    for idx, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 单页失败不影响其他页
            text = ""
        if text.strip():
            pages.append(f"【第 {idx} 页】\n{text.strip()}")

    joined = "\n\n".join(pages)
    if len(joined.strip()) < 40:
        raise ExtractError(
            "这份 PDF 没有文字层（多半是扫描件或拍照转的 PDF）。"
            "请改用「拍照 / 上传图片」，让模型直接读图。"
        )
    return joined


def _image_to_text(data: bytes) -> tuple[str, str]:
    """图片转文字。返回 (正文, 说明)，说明用于告诉用户用了哪个模型。"""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ExtractError("服务器缺少 Pillow，无法处理图片。") from exc

    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ExtractError("这张图片打不开，可能已损坏或格式不支持。") from exc

    # 压到最长边 1600，既保住可读性，也把 base64 体积控制住
    w, h = img.size
    if max(w, h) > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    raw = llm.chat_vision(b64, VISION_PROMPT, mime="image/jpeg")
    return _clean(_normalize_ocr(raw)), f"由 {config.VISION_MODEL} 识别"


def extract(filename: str, data: bytes) -> tuple[str, str]:
    """返回 (正文, 说明)。失败抛 ExtractError。"""
    if not data:
        raise ExtractError("文件是空的。")
    if len(data) > MAX_ATTACH_BYTES:
        raise ExtractError("文件超过 15MB，请压缩后再上传。")

    name = (filename or "").lower()
    suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""

    if suffix in TEXT_EXT:
        return _clean(_decode_text(data)), "已读取文本"

    if suffix in DOCX_EXT:
        return _clean(_from_docx(data)), "已解析 Word 文档"

    if suffix in PDF_EXT:
        return _clean(_from_pdf(data)), "已解析 PDF 文字层"

    if suffix in IMAGE_EXT:
        if not llm.available():
            raise ExtractError("图片需要模型识别，但当前没有配置模型密钥。")
        return _image_to_text(data)

    raise ExtractError(f"暂不支持 {suffix or '这种'} 格式。支持 txt / md / docx / pdf / 图片。")
