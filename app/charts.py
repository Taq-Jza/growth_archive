"""服务端生成图表。

不用 Chart.js 之类的库：一是又多一个外部依赖，二是这些图表很简单，
手写 SVG 反而更可控、首屏更快、也不会有 CDN 加载失败的风险。
"""
from __future__ import annotations

import math

# 明艳多色：色相拉开、明度与饱和对齐在同一档。
# 早期用过低饱和的莫兰迪色，投在白卡上会发灰、彼此难分；
# 这组更亮，相邻两色也不撞（青→橙→蓝→玫→绿→紫）。
PALETTE = ["#0FB5A8", "#F5A623", "#4C7DF5", "#F0567F", "#84C93C", "#8B5CF6"]

# 五类记录的颜色也从同一组取，避免「图表一套色、标签另一套色」
KIND_COLORS = {
    "course": PALETTE[0],
    "award": PALETTE[1],
    "skill": PALETTE[2],
    "project": PALETTE[3],
    "trait": PALETTE[4],
}

# 「其他」是兜底桶，不是能力域。用中性色，免得和一个真实能力域抢注意力，
# 也免得读者以为它和别的扇区是同一层级的维度。
OTHER_COLOR = "#B9B5AC"


def domain_colors(names: list[str]) -> dict[str, str]:
    """能力域 → 颜色，按映射表顺序固定分配。

    环形图和雷达图共用这一套：同一个域在两张图里必须同色，否则从概览点进
    放大视图时颜色会跳，读者要把两边的对应关系重新认一遍。
    """
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def _polar(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    """极坐标转直角坐标：0° 在 12 点、顺时针递增。"""
    a = math.radians(deg - 90)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _arc_cmds(cx: float, cy: float, r: float, start: float, end: float) -> str:
    """圆弧命令。超过 180° 就拆段 —— 单条 A 命令在起点与终点重合时不绘制，
    某一类占到 100% 时整圈会画不出来，拆段可以避开这个退化。"""
    span = end - start
    n = max(1, math.ceil(abs(span) / 179.9))
    sweep = 1 if span >= 0 else 0
    out = []
    for i in range(1, n + 1):
        x, y = _polar(cx, cy, r, start + span * i / n)
        out.append(f"A{r:.2f} {r:.2f} 0 0 {sweep} {x:.2f} {y:.2f}")
    return " ".join(out)


def _face_cmd(cx: float, cy: float, r_a: float, r_b: float, deg: float, bulge: float) -> str:
    """端面：从半径 r_a 走到 r_b，沿切向鼓出 bulge 度。

    两个控制点都落在端点的圆弧切线上，所以曲线与内外弧相切、不留折角；
    三次贝塞尔在 t=.5 处的偏移是控制点偏移的 0.75 倍，据此反推控制点长度，
    鼓包深度因此是精确值而不是目测值。
    """
    rad = math.radians(bulge)
    length = 4 * ((r_a + r_b) / 2) * abs(rad) / 3
    sign = 1 if rad >= 0 else -1
    ax, ay = _polar(cx, cy, r_a, deg)
    bx, by = _polar(cx, cy, r_b, deg)
    tx = math.cos(math.radians(deg)) * sign * length
    ty = math.sin(math.radians(deg)) * sign * length
    return (
        f"C{ax + tx:.2f} {ay + ty:.2f} {bx + tx:.2f} {by + ty:.2f} "
        f"{bx:.2f} {by:.2f}"
    )


def _slice_path(cx, cy, r_out, r_in, start, end, head_b, tail_b) -> str:
    """一段环带：内弧 → 尾端面（外凸）→ 外弧 → 头端面（内凹）→ 闭合。

    head_b / tail_b 是这个段两端接缝的鼓包深度。相邻两段在同一条接缝上取同一个
    值、调同一个函数，所以两条边严格重合 —— 这就是「无缝」的来源。
    """
    if abs(end - start) > 359.5:
        head_b = tail_b = 0.0   # 整圈时头尾重合，端面没有意义，退回完整的圆环
    x, y = _polar(cx, cy, r_in, start)
    return (
        f"M{x:.2f} {y:.2f} "
        + _arc_cmds(cx, cy, r_in, start, end) + " "
        + _face_cmd(cx, cy, r_in, r_out, end, tail_b) + " "
        + _arc_cmds(cx, cy, r_out, end, start) + " "
        + _face_cmd(cx, cy, r_out, r_in, start, head_b) + " Z"
    )


def donut(
    items: list[tuple[str, int]],
    size: int = 184,
    thickness: int = 28,
    keys: list[str] | None = None,
    colors: list[str] | None = None,
) -> str:
    """items: [(标签, 数值)]，返回完整 SVG 字符串。

    相邻两段无缝咬合的环形：
    - 每段的头部向内挖掉一个鼓包、尾部向外伸出同形状的一个；
    - 相邻两段共用同一条曲线（上一段由内向外走、下一段由外向内走），两条边
      严格重合，所以拼起来是一条连续带子，只靠颜色分段，中间没有缝；
    - 厚度恒定，读角度时不会被「粗细变化」干扰。

    极小切片应在调用方合并成「其他」：段数控制在 5–6 个，且每段要容得下鼓包，
    否则细段会被相邻两段咬穿。

    keys / colors 与 items 等长时，分别给每个扇区带上 data-key 与指定颜色。
    """
    total = sum(v for _, v in items)
    sw = float(thickness)
    cx = cy = size / 2.0
    r_out = size / 2.0 - 1        # 留 1px 余量，免得描边抗锯齿被 viewBox 切掉
    r_in = r_out - sw

    if total <= 0:
        return (
            f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
            f'<circle cx="{cx}" cy="{cy}" r="{(r_in + r_out) / 2:.2f}" '
            f'fill="none" stroke="#EEECE6" stroke-width="{sw:.1f}" /></svg>'
        )

    visible = [(i, label, value) for i, (label, value) in enumerate(items) if value > 0]
    sweeps = [360.0 * value / total for _, _, value in visible]
    # 每个接缝的鼓包深度由它两侧的段共同决定：取较短的一段，谁也咬不穿谁。
    # 再受环宽约束 —— 约环宽 0.35 倍的弧长看起来才像舌头，再长就成了尖刺。
    # 按接缝分别算而不是全局取最小值，否则一条只有两三个点的细分项会把
    # 整圈所有接缝的凹凸一起压平。
    r_mid = (r_out + r_in) / 2
    cap = math.degrees(0.35 * sw / r_mid)
    n = len(sweeps)
    seams = [min(cap, 0.42 * min(sweeps[j - 1], sweeps[j])) for j in range(n)] if n else []

    angle = 0.0
    segs: list[str] = []
    for i, ((idx, label, value), sweep) in enumerate(zip(visible, sweeps)):
        color = colors[idx] if colors and idx < len(colors) else PALETTE[idx % len(PALETTE)]
        key_attr = f' data-key="{keys[idx]}"' if keys and idx < len(keys) else ""
        pct = round(value / total * 100)
        d = _slice_path(cx, cy, r_out, r_in, angle, angle + sweep,
                        seams[i], seams[(i + 1) % n])
        segs.append(
            f'<path class="donut-seg" d="{d}" fill="{color}"{key_attr}'
            f' data-a="{angle:.3f}" data-b="{angle + sweep:.3f}"'
            f' data-hb="{seams[i]:.3f}" data-tb="{seams[(i + 1) % n]:.3f}">'
            f"<title>{label} · {value} 条 · {pct}%</title></path>"
        )
        angle += sweep

    # data-* 是给前端做入场动画用的：页面脚本按这些角度把路径从 0 重算一遍再张开，
    # 服务端仍然输出最终形状，所以没有脚本也能直接看到完整的环。
    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
        f'class="donut" data-rout="{r_out:.2f}" data-rin="{r_in:.2f}">'
        + "".join(segs) + "</svg>"
    )


def bars(items: list[tuple[str, int]], width: int = 260, row: int = 26) -> str:
    """横向条形图，用于类型分布。"""
    if not items:
        return ""
    peak = max(v for _, v in items) or 1
    height = row * len(items)
    parts: list[str] = []
    for idx, (_, value) in enumerate(items):
        y = idx * row
        w = max(1.0, (width - 60) * value / peak)
        color = PALETTE[idx % len(PALETTE)]
        parts.append(
            f'<rect x="0" y="{y + 6}" width="{w:.1f}" height="{row - 12}" rx="3" fill="{color}"/>'
        )
    return f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">' + "".join(parts) + "</svg>"


def ring(value: int, size: int = 116, thickness: int = 9) -> str:
    """环形进度条，用于展示匹配度这类单个百分比。带描边生长动画。"""
    value = max(0, min(100, int(value or 0)))
    r = (size - thickness) / 2
    circumference = 2 * math.pi * r
    offset = circumference * (1 - value / 100)

    # 分数越低越偏暖色，让「匹配度低」在视觉上也能被感知
    if value >= 70:
        color = "#0EA5A4"
    elif value >= 40:
        color = "#F59E0B"
    else:
        color = "#F43F5E"

    return f"""<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img">
  <circle cx="{size/2}" cy="{size/2}" r="{r:.1f}" fill="none"
          stroke="rgba(20,20,15,.07)" stroke-width="{thickness}"/>
  <circle class="ring-fg" cx="{size/2}" cy="{size/2}" r="{r:.1f}" fill="none"
          stroke="{color}" stroke-width="{thickness}" stroke-linecap="round"
          transform="rotate(-90 {size/2} {size/2})"
          style="--len:{circumference:.1f};--off:{offset:.1f}"/>
  <text x="50%" y="49%" text-anchor="middle" dominant-baseline="middle"
        font-size="{size*0.26:.0f}" fill="#14140F" font-weight="600">{value}</text>
  <text x="50%" y="68%" text-anchor="middle" dominant-baseline="middle"
        font-size="{size*0.09:.0f}" fill="rgba(20,20,15,.4)">匹配度</text>
</svg>"""
