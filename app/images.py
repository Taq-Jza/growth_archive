"""图片附件：原图保存、重绘、读取。

正文永远是检索与引用的依据，图片只是「看」的补充：
- original：用户上传的原图，保留笔迹、图示、证书等无法用文字表达的信息；
- redrawn：模型重排后的整洁笔记图，只用于阅读，不作为事实来源。

放在这里而不是塞进 importers，是因为写入/读取都涉及数据库与磁盘两侧状态，
和「把文件解析成文字」是两件事。
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import uuid
from pathlib import Path

from . import config, db, llm

UPLOAD_DIR = config.DATA_DIR / "uploads"

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# 重绘提示词：只允许「整理版式」，不允许增删内容。
# 实测该模型会把 ∇ 之类符号画丢，所以界面必须写明「文字以正文为准」。
REDRAW_PROMPT = """把这张学习笔记重绘成整洁的电子笔记：
1. 白底、左对齐、字号统一，行距均匀；
2. 保留原有的全部文字、公式、编号与结论，一个字都不要增删；
3. 公式里的符号按原样书写（如 ∇、θ、η、Σ），不要替换成方框或问号；
4. 不要添加标题装饰、边框、水印或任何原文没有的内容。"""


def _ensure_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_bytes(document_id: int, data: bytes, ext: str, role: str = "original", seq: int = 0) -> str:
    """把图片写盘并登记，返回可访问的相对路径。

    role=inline 时 seq 是它在正文里的序号，对应正文中的 ⟦IMG:n⟧ 占位。
    """
    _ensure_dir()
    ext = (ext or ".png").lower()
    if ext not in MIME_BY_EXT:
        ext = ".png"
    name = f"{uuid.uuid4().hex}{ext}"
    (UPLOAD_DIR / name).write_bytes(data)
    with db.get_conn() as conn:
        conn.execute(
            "INSERT INTO attachments (document_id, role, filename, mime, seq, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (document_id, role, name, MIME_BY_EXT[ext], int(seq or 0),
             dt.datetime.now().isoformat(timespec="seconds")),
        )
    return f"/media/{name}"


def list_for(document_id: int) -> list[dict]:
    rows = db.rows(
        "SELECT role, filename, mime, seq FROM attachments WHERE document_id = ? ORDER BY id",
        (document_id,),
    )
    return [
        {"role": r["role"], "url": f"/media/{r['filename']}", "mime": r["mime"], "seq": r["seq"]}
        for r in rows
    ]


# ---------------------------------------------------------------- 重排（默认）
FONT_CANDIDATES = [
    ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/PingFang.ttc"),
]


def _fonts(size: int = 26, title_size: int = 38):
    from PIL import ImageFont

    for body, bold in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(body, size), ImageFont.truetype(bold, title_size)
        except Exception:  # noqa: BLE001 换下一个候选
            continue
    base = ImageFont.load_default()
    return base, base


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in (text or "").splitlines():
        para = para.strip()
        if not para:
            lines.append("")
            continue
        cur = ""
        for ch in para:
            if draw.textlength(cur + ch, font=font) <= max_width:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


def typeset(text: str, title: str = "") -> bytes:
    """用识别出的文字重新排版成一张整洁的笔记图。

    为什么不用图生图模型：实测 Qwen-Image-Edit 会在中文上漏行、串字、把提示词
    当成正文画进去。既然文字已经识别得很准，直接排版既不会错，也只要几十毫秒。
    """
    from PIL import Image, ImageDraw

    width, pad = 1000, 56
    font, font_b = _fonts()

    probe = Image.new("RGB", (width, 10), "white")
    pdraw = ImageDraw.Draw(probe)
    body_lines = _wrap(pdraw, text.strip(), font, width - pad * 2)
    title_lines = _wrap(pdraw, title.strip(), font_b, width - pad * 2) if title.strip() else []

    line_h, para_gap = 44, 16
    height = pad + len(title_lines) * 56 + (24 if title_lines else 0) \
        + sum(para_gap if not ln else line_h for ln in body_lines) + pad
    height = max(460, height)

    img = Image.new("RGB", (width, height), "#FFFDF8")
    d = ImageDraw.Draw(img)

    # 顶部一条细色带，让图看起来是「一张卡片」而不是截图
    d.rectangle([0, 0, width, 6], fill="#0D9488")

    y = pad + 6
    for ln in title_lines:
        d.text((pad, y), ln, font=font_b, fill="#1A1D19")
        y += 56
    if title_lines:
        y += 18
        d.line([pad, y, width - pad, y], fill="#E7E3D8", width=2)
        y += 22

    for ln in body_lines:
        if not ln:
            y += para_gap
            continue
        d.text((pad, y), ln, font=font, fill="#2B2E28")
        y += line_h

    footer = dt.date.today().isoformat()
    d.text((width - pad - 96, height - 40), footer, font=font, fill="#B6B2A6")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def redraw(data: bytes, mime: str = "image/jpeg") -> bytes:
    """调用图生图模型，把原图重排成整洁笔记。失败抛 LLMUnavailable。"""
    img = _shrink(data)
    b64 = base64.b64encode(img).decode("ascii")
    return llm.edit_image(b64, REDRAW_PROMPT, mime="image/jpeg")


def _shrink(data: bytes, max_side: int = 1280) -> bytes:
    """压到 1280px 以内再送模型：既省流量，也避免超长边被拒。"""
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover
        return data
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue()
    except Exception:  # noqa: BLE001 图片坏了就按原样试一次
        return data
