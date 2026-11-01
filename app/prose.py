"""把结构化文本整理成自然语言正文。

为什么需要这一层：md / txt 是「排版稿」——`#` 是标题、`-` 是列表、`$...$` 是公式。
这些符号原样入库有三处坏处：
1. 列表摘要读起来是一片符号，用户看到的不是笔记而是源码；
2. 中文分词会把符号当边界，`$x=A\\cos(\\omega t+\\varphi)$` 这样的串检索命中率极低；
3. 思维导图、知识图谱都从正文抽结构，公式没转成普通文本，抽出来的节点就是
   「简谐振动方程：$x = A\\cos(\\omega t +」这种半截字。

处理分两级：
1. **本地清洗**（必做，毫秒级）：去排版符号、把常见 LaTeX 命令换成 Unicode 符号；
2. **模型规整**（可选）：按段改写成分段叙述，让公式以「T=2π√(m/k)」这类普通文本出现。

模型不可用、超时或返回空时，退回第 1 级结果 —— 至少可读，不是一片乱码。
两级都只做格式转换，不增删事实：数字、单位、结论、专有名词必须原样保留。
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from . import config, llm

# 单块送入模型的字符数。1600 字约 1200 token，四路并发时总耗时对长度不敏感。
BLOCK_CHARS = 900
MAX_BLOCKS = 24          # 约 1.9 万字；超出部分保留本地清洗结果，不截断原文
# 输出不得短于输入的这个比例：低于它说明模型在做概括而不是转写，
# 这时宁可退回本地清洗结果 —— 符号难看一点，但内容一条不少。
MIN_KEEP = 0.75
WORKERS = 6

PROMPT = """把下面这段课程资料原文转写成**可以直接阅读的中文笔记正文**。

这是「格式转换」，不是「总结」。硬性要求：
1. 原文有多少条知识点，输出就要有多少条：一条都不能合并、省略、概括、归并同类项。
   原文用短横线列出的每一项，输出里都要单独成句，不能并成一段。
2. 输出的信息量要与原文相当。去掉符号后字数会略少，但不得低于原文的八成 ——
   明显变短说明你做了概括，那是错的。
3. 去掉所有排版符号：标题井号、列表短横线、编号、加粗星号、表格竖线、代码反引号、引用尖角。
   章节名可以单独占一行，但不要带井号。
4. 公式转成普通文本，不用 LaTeX 记号：
   x = A\\cos(\\omega t + \\varphi) 写成  x = A·cos(ωt + φ)
   T = 2\\pi\\sqrt{m/k} 写成  T = 2π√(m/k)
   \\dfrac{1}{4\\pi\\varepsilon_0} 写成  1/(4πε₀)
   \\Delta\\varphi = \\pm2k\\pi 写成  Δφ = ±2kπ
   希腊字母用符号（ω φ λ μ π ε σ ρ θ Δ），乘号用 ×，点乘用 ·，分数用斜杠。
5. 原文的每个数字、单位、公式、结论、专有名词必须原样出现，不得改数值、不得补充原文没有的内容。
6. 不要写「本段介绍了」「以下是」这类话，直接从内容本身写起。
7. 原文是表格时，用句子把条目之间的关系讲清楚，不要还原成表格。
8. 只输出转写后的正文，不要任何解释、不要加标题。"""


# ---------------------------------------------------------------- 本地清洗

# LaTeX 命令 → 可读符号。顺序无关，逐个替换；未知命令统一去掉反斜杠。
_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "Sigma": "Σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Omega": "Ω",
    # 大写形式：模型偶尔写 \\Phi、\\Gamma，不映射会变成 "Phi" 这种半截词
    "Gamma": "Γ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Rho": "Ρ", "Tau": "Τ", "Phi": "Φ", "Psi": "Ψ", "Upsilon": "Υ",
    "pm": "±", "mp": "∓", "times": "×", "div": "÷", "cdot": "·", "cdots": "…",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠",
    "approx": "≈", "equiv": "≡", "propto": "∝", "infty": "∞", "partial": "∂",
    "sum": "Σ", "prod": "∏", "int": "∫", "oint": "∮", "oiint": "∬",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "circ": "°", "degree": "°", "angle": "∠", "perp": "⊥", "parallel": "∥",
    "in": "∈", "notin": "∉", "subset": "⊂", "cup": "∪", "cap": "∩",
    "emptyset": "∅", "forall": "∀", "exists": "∃", "therefore": "∴",
    "quad": " ", "qquad": "  ", "enspace": " ", "left": "", "right": "",
    "displaystyle": "", "limits": "", "nolimits": "",
}

_FUNCS = r"(cos|sin|tan|cot|sec|csc|arccos|arcsin|arctan|log|ln|lg|exp|max|min|lim|det)"


def _frac(m: "re.Match[str]") -> str:
    a, b = m.group(1), m.group(2)
    if re.fullmatch(r"[\w.]+", a) and re.fullmatch(r"[\w.]+", b):
        return f"{a}/{b}"
    return f"({a})/({b})"


def _sqrt(m: "re.Match[str]") -> str:
    inner = m.group(1)
    return f"√({inner})" if not re.fullmatch(r"[\w.]+", inner) else f"√{inner}"


def latex_to_text(text: str) -> str:
    """把 LaTeX 片段换成可读符号。

    只做常见命令的等价替换，不做完整解析：目的是「人能读、分词能切」，
    不是渲染出排版效果。遇到没见过的命令就只去掉反斜杠。
    """
    t = text

    # 相邻命令之间补一个空格：`\omega\sin` 展开后会粘成 `omegasin`，
    # 先断开才能在下一步各自替换成 ω 和 sin。
    t = re.sub(r"(\\(?:[A-Za-z]+))(?=\\)", r"\1 ", t)

    # 环境与对齐标记
    t = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", t)
    t = t.replace("\\\\", "；").replace("&", " ")

    # 字体包裹命令：\text{内} / \mathrm{d} 只保留内容，
    # 必须在分数与上下标之前做 —— `q_{text{内}}` 属于嵌套大括号，
    # 不先拆掉这层，下面的 `_{...}` 规则就匹配不到。
    t = re.sub(r"\\(?:text|textrm|mathrm|mathbf|mathit|mathbb|operatorname)\{([^{}]*)\}", r"\1", t)

    # 分数要在大括号还原之前处理，才能拿到分子分母的整体
    t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", _frac, t)
    # \frac12 是 \frac{1}{2} 的简写，理工资料里很常见
    t = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", r"\1/\2", t)
    t = re.sub(r"\\sqrt\{([^{}]*)\}", _sqrt, t)

    # 上标下标：x^{2} → x^2，v_{0} → v_0
    t = re.sub(r"\^\{([^{}]{1,3})\}", r"^\1", t)
    t = re.sub(r"_\{([^{}]{1,3})\}", r"_\1", t)

    # 函数名：\cos → cos
    t = re.sub(r"\\" + _FUNCS + r"\b", r"\1", t)

    # 向量与重音：\vec E → E，\hat x → x̂ 简化为 x
    t = re.sub(r"\\vec\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\vec\s*([A-Za-z])", r"\1", t)
    t = re.sub(r"\\hat\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\bar\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\overline\s*\{([^{}]*)\}", r"\1", t)

    # 希腊字母与其他符号
    def _sym(m: "re.Match[str]") -> str:
        name = m.group(1)
        return _SYMBOLS.get(name, name)

    t = re.sub(r"\\([A-Za-z]+)", _sym, t)

    # 符号替换完再过一遍上下标：`A_{\max}` 里的 `\max` 到这一步才变成 `max`，
    # 放在替换之前会因为带反斜杠而超出 {1,3} 的长度限制漏掉。
    t = re.sub(r"\^\{([^{}]{1,4})\}", r"^\1", t)
    t = re.sub(r"_\{([^{}]{1,4})\}", r"_\1", t)

    # 间距命令与残留反斜杠
    t = re.sub(r"\\[,;:! ]", " ", t)
    t = t.replace("\\", "")

    # 数学定界符
    t = t.replace("$$", " ").replace("$", "")
    # 希腊字母与后面的字母连写（Δ φ → Δφ、ω t → ωt），中文前保留空格
    t = re.sub(r"([Α-ωΔΣΩ∬∫∮]) +(?=[A-Za-zΑ-ω])", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t


def strip_markup(text: str) -> str:
    """去掉 Markdown / 文本里的排版符号，转成分段叙述。"""
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # 代码块围栏：保留内容、去掉 ```lang 行
    t = re.sub(r"^\s*```[^\n]*$", "", t, flags=re.M)
    t = re.sub(r"^\s*~~~[^\n]*$", "", t, flags=re.M)

    # 表格：| 项目 | 数值 | → 「项目：数值」
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
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)           # 标题井号
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)                 # 引用
    t = re.sub(r"^\s*[-*+]\s+\[[ xX]\]\s*", "", t, flags=re.M)  # 任务列表
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)            # 无序列表
    t = re.sub(r"^\s*\d+[.、)]\s+", "", t, flags=re.M)        # 有序列表
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)                    # 加粗
    t = re.sub(r"__(.+?)__", r"\1", t)                        # 加粗（下划线式）
    t = re.sub(r"(?<!\*)\*(?!\*)([^*\n]{1,60})(?<!\*)\*(?!\*)", r"\1", t)  # 斜体
    t = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", t)              # 行内代码
    t = re.sub(r"!\[([^\]]{0,80})\]\([^)]*\)", r"\1", t)      # 图片 → alt
    t = re.sub(r"\[([^\]]{0,80})\]\((https?://[^)]+)\)", r"\1（\2）", t)  # 链接
    t = re.sub(r"<[^>\n]{1,40}>", "", t)                       # 孤立 HTML 标签

    t = latex_to_text(t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------- 模型规整


def _split_blocks(text: str, limit: int = BLOCK_CHARS) -> list[str]:
    """按自然段打包成块，保证既不切断段落、块长又不超上限。

    只按空行切不够：Markdown 里 `-` 列出的十几行之间只有单个换行，
    整节会变成一个 1500 字的「段」，模型生成时间随之翻倍。
    所以超长段要再按行拆一次 —— 行是这里的最小单位，拆开也不会伤语义。
    """
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    paras: list[str] = []
    for p in raw_paras:
        if len(p) <= limit:
            paras.append(p)
            continue
        buf: list[str] = []
        size = 0
        for line in [ln for ln in p.split("\n") if ln.strip()]:
            if size + len(line) > limit and buf:
                paras.append("\n".join(buf))
                buf, size = [], 0
            buf.append(line)
            size += len(line)
        if buf:
            paras.append("\n".join(buf))

    blocks: list[str] = []
    buf = []
    size = 0
    for p in paras:
        if size + len(p) > limit and buf:
            blocks.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += len(p)
    if buf:
        blocks.append("\n\n".join(buf))
    return blocks


def _rewrite(block: str) -> str | None:
    try:
        out = llm.chat(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": block},
            ],
            temperature=0.15,
            max_tokens=2400,
            timeout=75,
        )
    except Exception:  # noqa: BLE001 单块失败不影响其余，退回原文
        return None
    out = (out or "").strip()
    # 两道闸门：空/拒答，以及「字数塌陷」。
    # 后者是实测出来的 —— 第一版提示词写「改写成自然语言」，模型理解成「总结」，
    # 5417 字压成 1953 字，公式还在但推导过程和分支条件没了。
    # 这种输出比原文更难用，宁可直接丢掉。
    if len(out) < max(20, len(block) * MIN_KEEP):
        return None
    return out


def to_prose(text: str, title: str = "") -> tuple[str, str]:
    """把排版稿整理成自然语言。返回 (正文, 说明)，说明直接展示给用户。"""
    cleaned = strip_markup(text)
    if not cleaned:
        return "", "文件里没有可读的文字"

    blocks = _split_blocks(cleaned)
    if not llm.available():
        return cleaned, "已提取文字（未配置模型，保留了原文的表格式写法）"

    head = blocks[:MAX_BLOCKS]
    tail = blocks[MAX_BLOCKS:]
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            done = list(ex.map(_rewrite, head))
    except Exception:  # noqa: BLE001 线程池异常整体降级
        return cleaned, "已提取文字（模型未响应，保留了原文的表格式写法）"

    ok = [d for d in done if d]
    parts: list[str] = []
    for orig, got in zip(head, done):
        parts.append(got if got else orig)
    # 超长文档的尾部不送模型，但也不丢：用本地清洗结果接在后面
    if tail:
        parts.extend(tail)
    body = "\n\n".join(p.strip() for p in parts if p and p.strip())

    total = len(done)
    if len(ok) == total and not tail:
        return body, f"已整理成自然语言（{config.CHAT_MODEL.rsplit('/', 1)[-1]}）"
    if not ok:
        return cleaned, "已提取文字（模型未响应，保留了原文的表格式写法）"
    miss = total - len(ok) + (1 if tail else 0)
    return body, f"已整理成自然语言，其中 {miss} 段因超时沿用原文"
