"""应用入口。路由全部是服务端渲染，不引前端框架。"""
from __future__ import annotations

import html as html_lib
import json
import math
import re
import datetime as dt
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import auth, cards, charts, config, db, embed, images, importers, ingest, interview, kg, llm, match, mindmap, prose, rag, seed, study, websearch
from .search import (
    browse,
    browse_count,
    kind_folders,
    capability_buckets,
    card_tag_folders,
    keyword_count,
    keyword_search,
    tag_folders,
)

_CAP_MAP_CACHE: dict | None = None


def _capability_map() -> dict:
    """能力域映射表，改 data/capability_map.json 即可调整，不用改代码。"""
    global _CAP_MAP_CACHE
    if _CAP_MAP_CACHE is None:
        path = config.BASE_DIR / "data" / "capability_map.json"
        try:
            _CAP_MAP_CACHE = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _CAP_MAP_CACHE = {}
    return _CAP_MAP_CACHE

PAGE_SIZE_HOME = 12
PAGE_SIZE_SEARCH = 10
PAGE_SIZE_CARDS = 24
KIND_ORDER = ["course", "skill", "award", "project", "trait"]


def kind_counts(user_id: int) -> list[tuple[str, int, int]]:
    """返回 [(kind, 中文名, 数量)]，按固定顺序。"""
    rows = db.rows(
        "SELECT kind, COUNT(*) AS n FROM documents WHERE user_id = ? GROUP BY kind",
        (user_id,),
    )
    found = {r["kind"]: r["n"] for r in rows}
    return [(k, ingest.KINDS[k], found.get(k, 0)) for k in KIND_ORDER]

app = FastAPI(title="成长档案")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
TEMPLATES.env.globals["KCOLORS"] = charts.KIND_COLORS
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)
# 原图与重绘图：落在 data/uploads，不在 static 里，避免被打包进版本库
(config.DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
app.mount(
    "/media",
    StaticFiles(directory=str(config.DATA_DIR / "uploads")),
    name="media",
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    auth.ensure_demo_user()


# ---------------- 统一错误页 ----------------
def _render_error(request: Request, code: int, title: str, detail: str):
    return TEMPLATES.TemplateResponse(
        request,
        "error.html",
        {"user": auth.current_user(request), "code": code, "title": title, "detail": detail},
        status_code=code,
    )


@app.exception_handler(StarletteHTTPException)
def http_error(request: Request, exc: StarletteHTTPException):
    mapping = {
        401: ("需要登录", "这个页面需要登录后访问。"),
        403: ("没有权限", "你没有访问这个页面的权限。"),
        404: ("页面不存在", "地址可能输错了，或者这条内容已经被删除。"),
        405: ("请求方式不对", "这个地址不支持当前操作。"),
    }
    title, detail = mapping.get(exc.status_code, ("请求出错了", "请稍后重试，或返回首页。"))
    return _render_error(request, exc.status_code, title, detail)


@app.exception_handler(Exception)
def server_error(request: Request, exc: Exception):  # noqa: ARG001
    return _render_error(
        request, 500, "服务出了点问题", "已经记录下来。可以先回首页，稍后再试一次。"
    )


def _set_session(resp: RedirectResponse, user_id: int) -> RedirectResponse:
    resp.set_cookie(
        auth.COOKIE_NAME,
        auth.make_cookie(user_id),
        httponly=True,
        samesite="lax",
        max_age=auth.MAX_AGE,
    )
    return resp


@app.get("/healthz")
def healthz():
    return {"ok": True, "db": str(config.DB_PATH)}


# ---------------- 认证 ----------------
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    """自己建账号。

    之前只有登录框没有注册入口，新用户拿到的是「请输入口令」——
    口令从哪来没人说，等于进不去。体验账号是给评委一键看效果的，
    不该是唯一入口。
    """
    if auth.current_user(request):      # 已经登录了就别再看注册页
        return RedirectResponse("/", status_code=303)
    return TEMPLATES.TemplateResponse(request, "register.html", {"error": None, "email": ""})


@app.post("/register")
def do_register(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    password2: str = Form(""),
):
    email = (email or "").strip()
    if password != password2:
        return TEMPLATES.TemplateResponse(
            request, "register.html", {"error": "两次输入的口令不一致", "email": email},
            status_code=400,
        )
    user, err = auth.register_user(email, password)
    if not user:
        return TEMPLATES.TemplateResponse(
            request, "register.html", {"error": err, "email": email}, status_code=400
        )
    return _set_session(RedirectResponse("/", status_code=303), user["id"])


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if auth.current_user(request):      # 已登录还停在登录页，多半是点了旧链接
        return RedirectResponse("/", status_code=303)
    return TEMPLATES.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
# 这些表单字段一律给默认空串，不写成 Form(...)。
# 声明为必填时，空提交会被 FastAPI 拦成 422 并落到通用错误页 ——
# 用户只看到「请求出错了」，不知道自己漏填了什么。
# 给默认值后走各自的业务校验，能回到页面说一句人话。
def do_login(request: Request, email: str = Form(""), password: str = Form("")):
    user = auth.login_user(email, password)
    if not user:
        return TEMPLATES.TemplateResponse(
            request, "login.html", {"error": "邮箱或口令不对"}, status_code=400
        )
    return _set_session(RedirectResponse("/", status_code=303), user["id"])


@app.get("/demo")
def demo_login():
    """一键体验：新用户不用注册。"""
    user = auth.login_user(config.DEMO_EMAIL, config.DEMO_PASSWORD)
    return _set_session(RedirectResponse("/", status_code=303), user["id"])


@app.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


# ---------------- 页面 ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request, page: int = 1, sort: str = "time"):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    sort = sort if sort in ("time", "kind") else "time"
    total_docs = db.one(
        "SELECT COUNT(*) AS n FROM documents WHERE user_id = ?", (user["id"],)
    )["n"]
    pages = max(1, math.ceil(total_docs / PAGE_SIZE_HOME))
    page = min(max(1, page), pages)
    offset = (page - 1) * PAGE_SIZE_HOME

    folders = kind_folders(user["id"]) if sort == "kind" else []
    docs = [] if folders else browse(user["id"], limit=PAGE_SIZE_HOME, offset=offset, sort=sort)
    for d in docs:
        d["kind_label"] = ingest.KINDS.get(d["kind"], d["kind"])

    cards = [
        dict(r)
        for r in db.rows(
            """SELECT c.*, d.title AS doc_title, d.kind AS kind
               FROM cards c JOIN documents d ON d.id = c.document_id
               WHERE c.user_id = ?
               ORDER BY COALESCE(d.period, d.created_at) DESC, c.id DESC
               LIMIT 6""",
            (user["id"],),
        )
    ]
    cards_total = db.one("SELECT COUNT(*) AS n FROM cards WHERE user_id = ?", (user["id"],))["n"]
    stats = {
        "docs": total_docs,
        "chunks": db.one(
            """SELECT COUNT(*) AS n FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE d.user_id = ?""",
            (user["id"],),
        )["n"],
        "cards": cards_total,
    }

    # 能力分布：主标签 → 能力域（映射表在 data/capability_map.json，可调）
    buckets = capability_buckets(
        user["id"], _capability_map(), tuple(ingest.KINDS.values())
    )

    # 环形图按记录类型分（课程 / 技能 / 竞赛 / 项目 / 特长）。
    #
    # 之前按能力域分，出了两个问题：能力域要先过标签映射，没标签或标签不在映射表里
    # 的记录只能落进「其他」；而占比 <5% 的小域又会被合并成另一个「其他」——
    # 于是图例里并排出现两个「其他」，各 2 条，谁也说不清哪个是哪个。
    # 记录类型是每条记录必有的属性，没有映射、不需要兜底，标签也是一眼能懂的词。
    kind_counts = {
        r["kind"]: r["n"]
        for r in db.rows(
            "SELECT kind, COUNT(*) AS n FROM documents WHERE user_id = ? GROUP BY kind",
            (user["id"],),
        )
    }
    kind_order = [k for k in ingest.KINDS if kind_counts.get(k)]
    if not kind_order:
        kind_order = list(ingest.KINDS)      # 空库时画一个灰环，不留白框

    pie_items = [(ingest.KINDS[k], kind_counts.get(k, 0)) for k in kind_order]
    pie_svg = charts.donut(
        pie_items, size=184, thickness=28, keys=list(kind_order),
        colors=[charts.KIND_COLORS[k] for k in kind_order],
    )

    # 图例点进去直接到该分类的列表页；这里没有「细分视图」，
    # 记录类型本身已经是最终粒度，再拆就只能拆到单条记录了。
    legend = [
        {
            "label": ingest.KINDS[k],
            "value": kind_counts.get(k, 0),
            "key": k,
            "color": charts.KIND_COLORS[k],
            "href": f"/c/{k}",
            "tags": [],
            "records": [],
            "merged": False,
        }
        for k in kind_order
    ]

    # 能力域配色按映射表顺序固定分配，雷达图用
    domains = [k for k in _capability_map() if not k.startswith("_")]
    dcolors = charts.domain_colors(domains)

    def color_of(name: str) -> str:
        return dcolors.get(name, charts.OTHER_COLOR)

    # 放大视图用六边形雷达图：轴必须固定，映射表里有几个能力域就画几条，
    # 没有记录的域也要出现 —— 0 值落在圆心，短板才看得见。
    # 顺序按记录数降序，优势在上、短板在下，扫一眼就知道强在哪、缺在哪。
    # 雷达轴要能点开细分视图，所以把该域下的标签构成与涉及记录一起带上。
    # 环形图改按记录类型分之后，这些字段就只有雷达图在用了 ——
    # 不带的话点轴进去是一片空白（openDetail 找不到 tags）。
    by_name = {b["name"]: b for b in buckets}
    spokes = sorted(
        (
            {
                "label": name,
                "value": by_name.get(name, {}).get("n", 0),
                "key": name,
                "color": color_of(name),
                "tags": by_name.get(name, {}).get("tags", []),
                "records": by_name.get(name, {}).get("records", []),
            }
            for name in domains
        ),
        key=lambda s: (-s["value"], s["label"]),
    )

    return TEMPLATES.TemplateResponse(
        request,
        "home.html",
        {
            "user": user,
            "docs": docs,
            "folders": folders,
            "cards": cards,
            "cards_total": cards_total,
            "cards_shown": len(cards),
            "stats": stats,
            "q": "",
            "page": page,
            "pages": pages,
            "total": total_docs,
            "sort": sort,
            "kinds": ingest.KINDS,
            "deleted": request.query_params.get("deleted", ""),
            "pie_svg": pie_svg,
            "legend": legend,
            "spokes": spokes,
            "has_data": total_docs > 0,
        },
    )


@app.get("/cards", response_class=HTMLResponse)
def cards_page(request: Request, kind: str = "", page: int = 1, sort: str = "time", tag: str = ""):
    """全部要点卡。

    概览页只放精选 6 张，这里按来源记录分组列出全部。
    每张卡标注它证明了什么能力 —— 简历与面试时按需取用。
    sort：time（最新）/ time_asc（最早）/ kind（按分类）。
    """
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    sort = sort if sort in ("time", "time_asc", "tag") else "time"
    kind = kind if kind in ingest.KINDS else ""
    where = "c.user_id = ?"
    params: list = [user["id"]]
    if kind:
        where += " AND d.kind = ?"
        params.append(kind)

    # 标签收藏夹：先看有哪些标签，点进去再看卡片
    folders: list[dict] = []
    if sort == "tag" and not tag:
        folders = card_tag_folders(user["id"], kind=kind)
    elif tag:
        labels = tuple(ingest.KINDS.values())
        label_ph = ",".join("?" for _ in labels)
        where += f""" AND (SELECT t2.name FROM document_tags dt2 JOIN tags t2 ON t2.id = dt2.tag_id
                            WHERE dt2.document_id = d.id AND t2.name NOT IN ({label_ph})
                            ORDER BY dt2.tag_id LIMIT 1) = ?"""
        params.extend(labels)
        params.append(tag)

    if folders:
        total = sum(f["n"] for f in folders)
    else:
        total = db.one(
            f"""SELECT COUNT(*) AS n FROM cards c JOIN documents d ON d.id = c.document_id
                WHERE {where}""",
            tuple(params),
        )["n"]
    pages = max(1, math.ceil(total / PAGE_SIZE_CARDS))
    page = min(max(1, page), pages)

    # 排序里必须带 d.id：否则同一文档的卡片可能被别的记录插开，分组就断了
    order = {
        "time_asc": "COALESCE(d.period, d.created_at) ASC, d.id, c.id",
        "kind": "d.kind, COALESCE(d.period, d.created_at) DESC, d.id, c.id",
    }.get(sort, "COALESCE(d.period, d.created_at) DESC, d.id, c.id")

    rows = [] if folders else db.rows(
        f"""SELECT c.*, d.title AS doc_title, d.kind AS kind, d.period AS period
            FROM cards c JOIN documents d ON d.id = c.document_id
            WHERE {where}
            ORDER BY {order}
            LIMIT ? OFFSET ?""",
        (*params, PAGE_SIZE_CARDS, (page - 1) * PAGE_SIZE_CARDS),
    )

    # 扁平列表 + 卡片自带来源：按记录分组再排两列时，只有一张卡的组右列会空着，
    # 视觉上像"漏排"。改成每张卡独立成块，来源信息放进卡片头部一角。
    cards = [dict(r) for r in rows]

    card_total = db.one(
        "SELECT COUNT(*) AS n FROM cards WHERE user_id = ?", (user["id"],)
    )["n"]
    card_counts = {
        r["kind"]: r["n"]
        for r in db.rows(
            """SELECT d.kind AS kind, COUNT(*) AS n
               FROM cards c JOIN documents d ON d.id = c.document_id
               WHERE c.user_id = ? GROUP BY d.kind""",
            (user["id"],),
        )
    }

    return TEMPLATES.TemplateResponse(
        request,
        "cards.html",
        {
            "user": user,
            "cards": cards,
            "total": total,
            "card_total": card_total,
            "card_counts": card_counts,
            "page": page,
            "pages": pages,
            "kind": kind,
            "sort": sort,
            "tag": tag,
            "folders": folders,
            "kinds": ingest.KINDS,
            "counts": kind_counts(user["id"]),
        },
    )


@app.get("/partial/records", response_class=HTMLResponse)
def partial_records(request: Request, page: int = 1, sort: str = "time"):
    """记录列表的局部刷新接口：只换列表，不重载整页。"""
    user = auth.current_user(request)
    if not user:
        return HTMLResponse("", status_code=401)

    sort = sort if sort in ("time", "kind") else "time"
    total_docs = db.one(
        "SELECT COUNT(*) AS n FROM documents WHERE user_id = ?", (user["id"],)
    )["n"]
    pages = max(1, math.ceil(total_docs / PAGE_SIZE_HOME))
    page = min(max(1, page), pages)

    # 按分类 = 文件夹层级，不再把所有记录按类摊开（记录一多就找不到）
    folders = kind_folders(user["id"]) if sort == "kind" else []
    docs = [] if folders else browse(
        user["id"], limit=PAGE_SIZE_HOME, offset=(page - 1) * PAGE_SIZE_HOME, sort=sort
    )
    for d in docs:
        d["kind_label"] = ingest.KINDS.get(d["kind"], d["kind"])

    return TEMPLATES.TemplateResponse(
        request,
        "_records.html",
        {
            "user": user,
            "docs": docs,
            "folders": folders,
            "page": page,
            "pages": pages,
            "total": total_docs,
            "sort": sort,
            "kinds": ingest.KINDS,
            "partial": True,
        },
    )


# 正在生成思维导图的记录 id。单进程部署，用进程内集合足够；
# 它只决定「页面上显示生成中还是空的」，重启丢一个标记不影响数据。
_MM_BUILDING: set[int] = set()


def _build_mindmap_bg(doc_id: int, user_id: int) -> None:
    try:
        mindmap.build(user_id, doc_id, force=True)
    finally:
        _MM_BUILDING.discard(doc_id)


@app.get("/mindmap/{doc_id}/state")
def mindmap_state(request: Request, doc_id: int):
    """给导图页轮询用：还在生成吗、出结果了吗。

    这里也要校验归属：只判登录的话，登录用户能用任意 id 探测
    别人有没有导图、是不是正在生成 —— 虽然拿不到内容，但没必要开这个口子。
    """
    user = auth.current_user(request)
    if not user:
        return {"building": False, "ready": False}
    if not db.one("SELECT id FROM documents WHERE id = ? AND user_id = ?", (doc_id, user["id"])):
        return {"building": False, "ready": False}
    return {
        "building": doc_id in _MM_BUILDING,
        "ready": mindmap.load(doc_id) is not None,
    }


@app.get("/mindmap/{doc_id}", response_class=HTMLResponse)
def mindmap_page(request: Request, background: BackgroundTasks, doc_id: int, rebuild: int = 0):
    """一份资料自己的思维导图：主题在一层、要点在二层，横向展开。

    生成一次要一分钟上下（四千多字的结构化 JSON），放在请求里同步等，
    浏览器先超时了。改成排队到后台，页面自己轮询状态。
    """
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    doc = db.one(
        "SELECT id, title, kind, period, created_at FROM documents WHERE id = ? AND user_id = ?",
        (doc_id, user["id"]),
    )
    if not doc:
        return RedirectResponse("/", status_code=303)

    tree = mindmap.load(doc_id)
    error = ""
    if rebuild:
        if not llm.available():
            error = "没有配置模型密钥，暂时生成不了思维导图。"
        elif doc_id not in _MM_BUILDING:
            _MM_BUILDING.add(doc_id)
            background.add_task(_build_mindmap_bg, doc_id, user["id"])
        return RedirectResponse(f"/mindmap/{doc_id}", status_code=303)

    if not tree and not llm.available():
        error = "没有配置模型密钥，暂时生成不了思维导图。"

    building = doc_id in _MM_BUILDING
    chart = mindmap.layout(tree) if tree else None
    return TEMPLATES.TemplateResponse(
        request,
        "mindmap.html",
        {
            "user": user,
            "doc": doc,
            "tree": tree,
            "chart": chart,
            "error": error,
            "building": building,
            "kinds": ingest.KINDS,
        },
    )


@app.get("/kg", response_class=HTMLResponse)
def kg_page(
    request: Request, kind: str = "", tag: str = "", built: int = 0, ok: int = 0, fail: int = 0,
    err: str = "",
):
    """某个标签下的知识图谱：章节在内圈，记录在外圈，点记录直接打开。"""
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    kind = kind if kind in ingest.KINDS else "course"
    tag = tag.strip()

    # 先看这个标签下到底有没有记录：rows 为空有两种原因 ——
    # 标签下没有记录（该去补标签），或记录还没生成过章节（该点生成），
    # 不区分开，用户只会看到同一句「还没有生成结构」。
    scope = kg.docs_in_tag(user["id"], kind, tag or None)
    rows = kg.load(user["id"], kind, tag or None)
    chapters = len({r["chapter"] for r in rows})
    seen: list[str] = []
    for r in rows:
        for c in r["concepts"]:
            if c not in seen:
                seen.append(c)

    # 课程/标签选择器：把「哪几门课」说清楚，别把所有课程混在一张图上。
    # 当前标签若排在第 12 名之后，要单独补进来，否则页面上看不出自己站在哪。
    folders = tag_folders(user["id"], kind=kind)
    head = folders[:12]
    if tag and all(f["name"] != tag for f in head):
        cur = next((f for f in folders if f["name"] == tag), {"name": tag, "n": len(scope)})
        head = head + [cur]
    folders = head

    return TEMPLATES.TemplateResponse(
        request,
        "kg.html",
        {
            "user": user,
            "kind": kind,
            "tag": tag,
            "center": tag or ingest.KINDS[kind],
            "graph_label": ingest.GRAPH_LABEL.get(kind, "知识图谱"),
            "rows": rows,
            "scope_total": len(scope),
            "chapters": chapters,
            "concepts": seen[:24],
            "graph": kg.layout(rows) if rows else None,
            "folders": folders,
            "built": {"ok": ok, "failed": fail, "skipped": max(0, len(scope) - ok - fail)}
            if built else None,
            "build_error": err,
            "kinds": ingest.KINDS,
        },
    )


@app.post("/kg/build")
def kg_build(request: Request, kind: str = Form("course"), tag: str = Form(""), force: int = Form(0)):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    kind = kind if kind in ingest.KINDS else "course"
    tag = tag.strip()

    try:
        res = kg.build(user["id"], kind, tag or None, force=bool(force))
    except llm.LLMUnavailable as exc:
        return RedirectResponse(
            f"/kg?kind={kind}&tag={quote(tag)}&err={quote('模型暂不可用：' + str(exc)[:60])}",
            status_code=303,
        )
    return RedirectResponse(
        f"/kg?kind={kind}&tag={quote(tag)}&built=1&ok={res['ok']}&fail={res['failed']}",
        status_code=303,
    )


@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = "", kind: str = "", frm: str = "", to: str = "", tag: str = "", page: int = 1):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    kind = kind if kind in ingest.KINDS else ""
    has_filter = bool(q.strip() or kind or frm or to or tag)

    # 用类别名搜索时直接进对应分类页：用户要的是「这个类别下的记录」，
    # 而不是正文里恰好出现「课程」两个字的片段。
    label_to_kind = {v: k for k, v in ingest.KINDS.items()}
    if q.strip() in label_to_kind and not (kind or frm or to or tag):
        return RedirectResponse(f"/c/{label_to_kind[q.strip()]}", status_code=303)

    if q.strip():
        total = keyword_count(user["id"], q, kind=kind or None, period_from=frm or None, period_to=to or None)
        if tag:
            total = browse_count(
                user["id"], kind=kind or None, period_from=frm or None, period_to=to or None, tag=tag
            )
    elif has_filter:
        # 只筛选不搜索也允许，见 AC-03-4 的例外条款
        total = browse_count(
            user["id"], kind=kind or None, period_from=frm or None, period_to=to or None, tag=tag or None
        )
    else:
        total = 0

    pages = max(1, math.ceil(total / PAGE_SIZE_SEARCH))
    page = min(max(1, page), pages)
    offset = (page - 1) * PAGE_SIZE_SEARCH

    if q.strip() and not tag:
        hits = keyword_search(
            user["id"], q, limit=PAGE_SIZE_SEARCH, offset=offset,
            kind=kind or None, period_from=frm or None, period_to=to or None,
        )
    elif has_filter:
        hits = browse(
            user["id"], kind=kind or None, period_from=frm or None, period_to=to or None,
            limit=PAGE_SIZE_SEARCH, offset=offset, tag=tag or None,
        )
    else:
        hits = []

    return TEMPLATES.TemplateResponse(
        request,
        "search.html",
        {
            "user": user,
            "q": q,
            "hits": hits,
            "kinds": ingest.KINDS,
            "kind": kind,
            "frm": frm,
            "to": to,
            "tag": tag,
            "page": page,
            "pages": pages,
            "total": total,
        },
    )


@app.get("/doc/{doc_id}", response_class=HTMLResponse)
def doc_page(request: Request, doc_id: int, cid: int | None = None, card_msg: str = "", card_ok: int = 0,
             saved: int = 0, quick: int = 0, note: str = "", img: str = "", why: str = "",
             hl: str = "", origin: str = Query("", alias="from")):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    doc = db.one(
        "SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user["id"])
    )
    if not doc:
        return RedirectResponse("/", status_code=303)
    tags = db.rows(
        """SELECT t.name FROM tags t JOIN document_tags dt ON dt.tag_id = t.id
           WHERE dt.document_id = ?""",
        (doc_id,),
    )
    # 「返回」要回到用户来的地方。从要点卡点进来却跳回概览，
    # 等于把人丢回起点，还得重新点一遍卡片。来处由入口链接用 ?from= 带过来，
    # 白名单之外一律按概览处理，免得被外部链接塞任意地址。
    origin = (origin or "").strip().lower()
    if origin == "cards":
        back_href, back_label = "/cards", "返回要点卡"
    elif origin == "search":
        back_href, back_label = "/search", "返回检索"
    elif origin == "interview":
        back_href, back_label = "/interview", "返回面试预演"
    elif origin == "category":
        back_href, back_label = f"/c/{doc['kind']}", f"返回{ingest.KINDS.get(doc['kind'], '列表')}"
    else:
        back_href, back_label = "/", "返回概览"

    cards = db.rows("SELECT * FROM cards WHERE document_id = ?", (doc_id,))
    files = images.list_for(doc_id)
    has_redraw = any(f["role"] == "redrawn" for f in files)
    inline = {f["seq"]: f["url"] for f in files if f["role"] == "inline" and f["seq"]}

    # 引用跳转：把被引用的片段在正文中高亮出来（对应 AC-05-2）
    body_html = html_lib.escape(doc["body"])
    if cid:
        chunk = db.one(
            "SELECT text FROM chunks WHERE id = ? AND document_id = ?", (cid, doc_id)
        )
        if chunk:
            needle = html_lib.escape(chunk["text"])
            if needle and needle in body_html:
                body_html = body_html.replace(
                    needle, f'<mark id="cited">{needle}</mark>', 1
                )

    # 思维导图点节点会带 ?hl=原文片段：把这一段标出来并滚到它那儿。
    # 片段来自模型摘录，可能与正文差一个空格或标点，所以按空白宽松匹配，
    # 找不到就退化成「只按前 10 个字找」，再不行就当普通打开、不报错。
    anchored = False
    if hl.strip():
        plain = html_lib.escape(hl.strip()[:40])
        pattern = r"\s+".join(re.escape(p) for p in plain.split())
        m = re.search(pattern, body_html)
        if not m and len(plain) > 10:
            head = html_lib.escape(hl.strip()[:10])
            m = re.search(r"\s+".join(re.escape(p) for p in head.split()), body_html)
        if m:
            body_html = (
                body_html[: m.start()]
                + '<mark id="cited" class="doc-cited">'
                + m.group(0)
                + "</mark>"
                + body_html[m.end():]
            )
            anchored = True

    # 文档内嵌图：正文里的 ⟦IMG:n⟧ 换成真实图片。
    # 在 escape 之后替换 —— 占位符全是普通字符，不会被转义，替换进去的 img 标签才不会被二次转义。
    body_html = importers.IMG_MARK_RE.sub(
        lambda m: (
            f'<img class="doc-inline-img" src="{inline[int(m.group(1))]}" '
            f'alt="图 {m.group(1)}" loading="lazy">'
            if int(m.group(1)) in inline
            else '<span class="doc-img-miss">（这张图没能保存下来）</span>'
        ),
        body_html,
    )
    # 粘贴进来的 Markdown 图片：保留外链，不下载也不改写
    body_html = importers.MD_IMG_RE.sub(
        lambda m: (
            f'<img class="doc-inline-img" src="{m.group(2)}" '
            f'alt="{m.group(1) or "图片"}" loading="lazy">'
        ),
        body_html,
    )

    return TEMPLATES.TemplateResponse(
        request,
        "doc.html",
        {
            "user": user,
            "doc": doc,
            "tags": tags,
            "cards": cards,
            "kinds": ingest.KINDS,
            "body_html": body_html,
            "card_msg": card_msg,
            "card_ok": card_ok,
            "saved": saved,
            "quick": quick,
            "note": note,
            "back_href": back_href,
            "back_label": back_label,
            "files": files,
            "has_redraw": has_redraw,
            "img_state": img,
            "img_msg": why,
        },
    )


@app.post("/card/{card_id}/delete")
def delete_card(request: Request, card_id: int):
    """删掉单张要点卡。

    卡片是整条记录的产物，但一条记录提 3 张卡里常常只有 1 张不合用 ——
    为删一张而重提整批太重。删卡不动记录本身，也不动其他卡。
    """
    user = auth.current_user(request)
    if not user:
        if _wants_json(request):
            return JSONResponse({"ok": False, "error": "未登录"}, status_code=401)
        return RedirectResponse("/login", status_code=303)
    # 先查再删：db.execute 返回的是 lastrowid，DELETE 拿不到「删了几行」，
    # 用它判断成功与否永远是 False —— 表现是接口一直回 404、前端提示删除失败。
    if not db.one("SELECT id FROM cards WHERE id = ? AND user_id = ?", (card_id, user["id"])):
        if _wants_json(request):
            return JSONResponse({"ok": False, "error": "没有这张卡"}, status_code=404)
        return RedirectResponse("/cards", status_code=303)
    db.execute("DELETE FROM cards WHERE id = ? AND user_id = ?", (card_id, user["id"]))
    if _wants_json(request):
        return JSONResponse({"ok": True})
    return RedirectResponse("/cards", status_code=303)


@app.post("/doc/{doc_id}/cards")
def generate_cards(request: Request, doc_id: int):
    """要点提炼（F-06）。失败时把原因带回详情页，不吞错误。"""
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    result = cards.extract(user["id"], doc_id)
    if "cards" in result:
        return RedirectResponse(f"/doc/{doc_id}?card_ok=1", status_code=303)
    return RedirectResponse(
        f"/doc/{doc_id}?card_msg={quote(result.get('error', '提炼失败'))}", status_code=303
    )


# ---------------- 问答 ----------------
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/ask", response_class=HTMLResponse)
def ask_page(request: Request, q: str = "", scope: str = ""):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    scope = scope if scope in ingest.KINDS else ""
    return TEMPLATES.TemplateResponse(
        request,
        "ask.html",
        {
            "user": user,
            "q": q,
            "scope": scope,
            "scope_label": ingest.KINDS.get(scope, ""),
            "llm_ready": rag.llm.available(),
            "embed_ready": embed.available(),
        },
    )


@app.get("/api/ask")
def api_ask(request: Request, q: str = "", scope: str = ""):
    user = auth.current_user(request)
    if not user:
        return StreamingResponse(
            iter([_sse("fail", {"message": "未登录"})]), media_type="text/event-stream"
        )
    scope = scope if scope in ingest.KINDS else ""

    def gen():
        try:
            for event, payload in rag.answer_stream(user["id"], q, kind=scope or None):
                yield _sse(event, payload)
        except Exception as exc:  # noqa: BLE001
            yield _sse("fail", {"message": f"服务异常：{str(exc)[:120]}"})
        yield _sse("close", {})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/new", response_class=HTMLResponse)
def new_page(request: Request, kind: str = "", err: str = ""):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return TEMPLATES.TemplateResponse(
        request,
        "new.html",
        {
            "user": user,
            "kinds": ingest.KINDS,
            "preset_kind": kind if kind in ingest.KINDS else "course",
            "error": err,
            "vision_ready": embed.available(),
        },
    )


def _ocr_inline_images(doc_id: int, items: list[tuple[int, bytes]]) -> None:
    """把文档内嵌图的文字识别出来，插回正文对应位置。

    20 张图串行要两三分钟，放在上传请求里干等不合理：先让用户看到图，
    文字在后台慢慢补，补完检索就能命中图里的内容。
    并发 4 路是实测的折中 —— 再高对端限流，再低等太久。
    """
    if not items or not llm.available():
        return
    from concurrent.futures import ThreadPoolExecutor

    def one(pair: tuple[int, bytes]) -> tuple[int, str]:
        seq, data = pair
        try:
            text, _ = importers.image_to_text(data)
            return seq, text
        except Exception:  # noqa: BLE001 单张失败就跳过，不影响其它图
            return seq, ""

    got: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for seq, text in pool.map(one, items):
            if text:
                got[seq] = text

    if not got:
        return
    row = db.one("SELECT body FROM documents WHERE id = ?", (doc_id,))
    if not row:
        return
    body = row["body"]
    for seq, text in got.items():
        mark = importers.IMG_MARK.format(n=seq)
        # 图注紧跟占位：图和它的文字算同一段，检索命中时能一起返回
        body = body.replace(mark, f"{mark}\n（图 {seq} 内容）{text}", 1)
    ingest.refresh_body(doc_id, body)


def _polish_text(doc_id: int, raw: str) -> None:
    """后台把排版稿（Markdown / 纯文本）转写成自然语言。

    为什么不在上传请求里做：三千字的 Markdown 分三块并发调模型也要 30 秒，
    让用户在上传页干等不如先给一版本地清洗的正文 —— 符号已经去掉、检索已经能用，
    模型版改完直接替换正文并重建索引，刷新即可看到。
    """
    if not llm.available():
        return
    row = db.one("SELECT title, body FROM documents WHERE id = ?", (doc_id,))
    if not row:
        return
    body, _note = prose.to_prose(raw, row["title"])
    if not body.strip() or body.strip() == (row["body"] or "").strip():
        return
    ingest.refresh_body(doc_id, body)


@app.post("/upload")
async def upload_file(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile | None = File(None),
    kind: str = Form("course"),
    title: str = Form(""),
    period: str = Form(""),
    tags: str = Form(""),
    image_mode: str = Form("original"),
):
    """文件 / 图片导入。

    图片有两种处理方式：
    - original：保留原图，正文只取识别出的文字（笔迹、图示、证书原件都在图里）；
    - redraw：额外请模型重排一张整洁笔记图，原图同样保留，随时可切回。
    重绘失败不影响入库 —— 文字与原图都已经拿到，只是少一张「好看版」。
    """
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if kind not in ingest.KINDS:
        kind = "course"

    # 没选文件就点「上传并解析」：前端 reportValidity() 会拦一道，这里再兜一层。
    # 表单字段声明成必填的话，空提交会先被 FastAPI 判成 422、掉进通用错误页，
    # 用户看不到「到底哪一步没做」。
    if file is None or not (file.filename or "").strip():
        return RedirectResponse(
            f"/new?kind={kind}&err={quote('请先选择要上传的文件')}", status_code=303
        )

    data = await file.read()
    try:
        text, note, pics, pending = importers.extract(file.filename or "", data)
    except importers.ExtractError as exc:
        return RedirectResponse(
            f"/new?kind={kind}&err={quote(str(exc))}", status_code=303
        )

    if len(text.strip()) < 20:
        return RedirectResponse(
            f"/new?kind={kind}&err={quote('解析出来的内容太少，可能是空文件或图片太模糊。')}",
            status_code=303,
        )

    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if not title.strip():
        stem = (file.filename or "导入的资料").rsplit(".", 1)[0]
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        # 手机拍的照片文件名是 IMG_1234，没有信息量；这时用识别出的首行当标题
        if suffix in importers.IMAGE_EXT and 4 <= len(first) <= 32:
            title = first
        else:
            title = (stem or first)[:80]

    doc_id = ingest.ingest_document(
        user["id"], title, text, kind=kind, period=period.strip() or None, source_kind="file"
    )
    # 标签原来只走「粘贴文本」那条路，上传文件永远没有标签 ——
    # 后果是按标签分组的收藏夹、知识图谱、按标签检索全都看不到这些记录。
    if tags.strip():
        ingest.add_tags(doc_id, user["id"], [t for t in tags.replace("，", ",").split(",")])

    extra = ""
    if pending and pending.get("raw"):
        # 文本类资料的「转写成自然语言」放到后台：本地清洗版已经在库里、能检索，
        # 模型版改完直接替换正文并重建索引，用户不用在上传页等半分钟。
        background.add_task(_polish_text, doc_id, pending["raw"])
    if suffix in importers.IMAGE_EXT:
        images.save_bytes(doc_id, data, suffix, role="original")
        if image_mode == "redraw":
            # 默认用文字重新排版：一字不改、几十毫秒；IMAGE_REDRAW=ai 时才走图生图模型
            try:
                if config.IMAGE_REDRAW == "ai":
                    redrawn = images.redraw(data)
                else:
                    redrawn = images.typeset(text, title)
                images.save_bytes(doc_id, redrawn, ".png", role="redrawn")
                extra = "&img=redraw"
            except Exception as exc:  # noqa: BLE001 重排失败不影响入库
                extra = "&img=fail&why=" + quote(f"重排未成功（{str(exc)[:40]}），已保留原图")
        else:
            extra = "&img=original"
    elif pics:
        # 文档里的内嵌图：按正文序号落盘，详情页据此把 ⟦IMG:n⟧ 换成真图。
        # 图本身不参与检索（正文才是依据），但图里的文字要 —— 交给后台识别后补进正文。
        saved: list[tuple[int, bytes]] = []
        for p in pics:
            try:
                images.save_bytes(doc_id, p["data"], p["ext"], role="inline", seq=p["index"])
                saved.append((p["index"], p["data"]))
            except Exception:  # noqa: BLE001 单张失败不影响其余
                continue
        if saved:
            extra = f"&img=inline&n={len(saved)}"
            background.add_task(_ocr_inline_images, doc_id, saved)
            if llm.available():
                note += "，图片里的文字正在后台识别"
    return RedirectResponse(f"/doc/{doc_id}?note={quote(note)}{extra}", status_code=303)


@app.post("/new")
def new_document(
    request: Request,
    title: str = Form(""),
    body: str = Form(""),
    kind: str = Form("course"),
    period: str = Form(""),
    tags: str = Form(""),
):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    if kind not in ingest.KINDS or not body.strip():
        return TEMPLATES.TemplateResponse(
            request,
            "new.html",
            {"user": user, "kinds": ingest.KINDS, "error": "请选择记录类型，并填写非空正文"},
            status_code=400,
        )

    doc_id = ingest.ingest_document(
        user["id"],
        title,
        body,
        kind=kind,
        period=period.strip() or None,
        source_kind="paste",
    )
    ingest.add_tags(doc_id, user["id"], [t for t in tags.replace("，", ",").split(",")])
    return RedirectResponse(f"/doc/{doc_id}", status_code=303)


@app.get("/doc/{doc_id}/edit", response_class=HTMLResponse)
def edit_page(request: Request, doc_id: int):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    doc = db.one("SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user["id"]))
    if not doc:
        return RedirectResponse("/", status_code=303)
    return TEMPLATES.TemplateResponse(
        request,
        "edit.html",
        {
            "user": user,
            "doc": doc,
            "kinds": ingest.KINDS,
            "tag_text": "，".join(ingest.doc_tags(doc_id)),
        },
    )


@app.post("/doc/{doc_id}/edit")
def edit_submit(
    request: Request,
    doc_id: int,
    title: str = Form(""),
    body: str = Form(""),
    kind: str = Form(""),
    period: str = Form(""),
    tags: str = Form(""),
):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    if kind not in ingest.KINDS or not body.strip():
        doc = db.one("SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user["id"]))
        return TEMPLATES.TemplateResponse(
            request,
            "edit.html",
            {
                "user": user,
                "doc": doc,
                "kinds": ingest.KINDS,
                "tag_text": tags,
                "error": "类型必选，正文不能为空",
            },
            status_code=400,
        )

    ok = ingest.update_document(
        user["id"], doc_id, title, body, kind, period.strip() or None
    )
    if not ok:
        return RedirectResponse("/", status_code=303)
    ingest.set_tags(doc_id, user["id"], [t for t in tags.replace("，", ",").split(",")])
    return RedirectResponse(f"/doc/{doc_id}?saved=1", status_code=303)


@app.get("/interview", response_class=HTMLResponse)
def interview_index(request: Request, sort: str = "time", tag: str = ""):
    """面试预演的入口：先挑一条记录，再开始预演。

    这个功能一直存在（`/doc/{id}/interview`），但只有记录详情页右上角一个小按钮，
    侧栏没有入口 —— 用的人自然以为它被删了。

    排序与分类页同一套口径（最新 / 最早 / 按标签）：
    记录一多，一个平铺列表就只是在滚，找不到想练的那一条。
    """
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    if sort not in ("time", "time_asc", "tag"):
        sort = "time"
    tag = tag.strip()

    folders: list[dict] = []
    if sort == "tag" and not tag:
        folders = tag_folders(user["id"])
        docs: list[dict] = []
    else:
        docs = browse(
            user["id"], limit=60, tag=tag or None,
            sort="time" if sort == "tag" else sort,
        )

    return TEMPLATES.TemplateResponse(
        request,
        "interview_list.html",
        {
            "user": user,
            "docs": docs,
            "folders": folders,
            "sort": sort,
            "tag": tag,
            "total": len(docs),
            "kinds": ingest.KINDS,
            "web_ready": websearch.available(),
            "web_provider": config.SEARCH_PROVIDER,
        },
    )


@app.get("/doc/{doc_id}/interview", response_class=HTMLResponse)
def interview_page(request: Request, doc_id: int):
    """预演页。有上次的结果就直接摆出来，不让人干等 20 秒。"""
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    doc = db.one("SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user["id"]))
    if not doc:
        return RedirectResponse("/", status_code=303)
    cached = interview.load_cached(user["id"], doc)
    if cached:
        cached["cached"] = True
    return TEMPLATES.TemplateResponse(
        request,
        "interview.html",
        {"user": user, "doc": doc, "result": cached, "error": None},
    )


@app.post("/doc/{doc_id}/interview", response_class=HTMLResponse)
def interview_run(request: Request, doc_id: int, force: int = Form(0)):
    """生成预演题目。force=1 走「换一批」，忽略缓存重新生成。"""
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    doc = db.one("SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user["id"]))
    if not doc:
        return RedirectResponse("/", status_code=303)

    res = interview.prepare(user["id"], doc_id, force=bool(force))
    return TEMPLATES.TemplateResponse(
        request,
        "interview.html",
        {
            "user": user,
            "doc": doc,
            "result": None if "error" in res else res,
            "error": res.get("error"),
        },
    )


def _wants_json(request: Request) -> bool:
    """前端用 fetch 调接口时带这个头，用来区分「页面表单提交」和「脚本调用」。

    同一个接口两种出口：从列表右键删除，原地移掉那张卡、不刷新页面；
    从编辑页的按钮走普通表单，仍然整页跳回概览。
    """
    return request.headers.get("x-requested-with", "").lower() == "fetch"


@app.post("/doc/{doc_id}/delete")
def delete_document(request: Request, doc_id: int):
    user = auth.current_user(request)
    if not user:
        if _wants_json(request):
            return JSONResponse({"ok": False, "error": "未登录"}, status_code=401)
        return RedirectResponse("/login", status_code=303)
    ok = ingest.delete_document(user["id"], doc_id)
    if _wants_json(request):
        return JSONResponse({"ok": ok}, status_code=200 if ok else 404)
    return RedirectResponse("/?deleted=1", status_code=303)


@app.post("/docs/bulk-delete")
async def bulk_delete(request: Request):
    """列表里多选后一次删掉。

    逐条走 delete_document（而不是一条 SQL 删多行），是因为它还要清切片、
    删磁盘图片、让向量缓存失效 —— 绕开它就会留下孤儿文件。
    """
    user = auth.current_user(request)
    if not user:
        return JSONResponse({"ok": False, "error": "未登录"}, status_code=401)
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 空 body / 非 JSON 都当没选
        payload = {}
    ids = []
    for raw in (payload.get("ids") or [])[:100]:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    done = sum(1 for i in ids if ingest.delete_document(user["id"], i))
    return JSONResponse({"ok": True, "deleted": done})


# ---------------- 快速捕获 ----------------
@app.post("/quick")
def quick_capture(request: Request, text: str = Form(""), kind: str = Form("course")):
    """一句话先记下来，细节稍后补。降低录入门槛的关键。"""
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    text = text.strip()
    if not text:
        return RedirectResponse("/", status_code=303)
    if kind not in ingest.KINDS:
        kind = "course"

    title = text.splitlines()[0][:40] or "未命名记录"
    doc_id = ingest.ingest_document(
        user["id"], title, text, kind=kind, period=dt.date.today().isoformat(),
        source_kind="paste",
    )
    return RedirectResponse(f"/doc/{doc_id}?quick=1", status_code=303)


# ---------------- 分类页 ----------------
@app.get("/c/{kind}", response_class=HTMLResponse)
def category_page(
    request: Request, kind: str, page: int = 1, tag: str = "", sort: str = "time", q: str = ""
):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    if kind not in ingest.KINDS:
        return _render_error(request, 404, "分类不存在", "没有这个分类，回概览页看看。")

    sort = sort if sort in ("time", "time_asc", "tag") else "time"

    # 「按标签」且没选中具体标签时，先给一层收藏夹，点进去再看记录。
    # 直接把所有记录按标签排序铺开，等于没有分类，标签之间没有边界感。
    q = q.strip()
    folders: list[dict] = []
    # 搜索时直接落到记录列表，否则会出现「搜了关键词却只看到文件夹」
    if sort == "tag" and not tag and not q:
        folders = tag_folders(user["id"], kind=kind)

    # 页头的「N 条记录」永远是这一类别的总量；total 只用于当前筛选结果，
    # 否则一搜索，页头数字也跟着变小，看起来像记录丢了。
    all_total = browse_count(user["id"], kind=kind)
    total = browse_count(user["id"], kind=kind, tag=tag or None, q=q or None)
    pages = max(1, math.ceil(total / PAGE_SIZE_HOME))
    page = min(max(1, page), pages)
    offset = (page - 1) * PAGE_SIZE_HOME

    items = (
        []
        if folders
        else browse(
            user["id"], kind=kind, limit=PAGE_SIZE_HOME, offset=offset,
            tag=tag or None, sort=sort, q=q or None,
        )
    )

    label_set = list(ingest.KINDS.values())
    ph = ",".join("?" for _ in label_set)
    # 计数与筛选同口径：数「挂着这个标签的记录数」，筛选也按任意标签命中。
    # 早先两边都只认主标签，数字能对上，但次标签在界面上等于不存在。
    tag_rows = db.rows(
        f"""SELECT t.name AS name, COUNT(DISTINCT d.id) AS n
            FROM tags t
            JOIN document_tags dt ON dt.tag_id = t.id
            JOIN documents d ON d.id = dt.document_id
            WHERE d.user_id = ? AND d.kind = ?
              AND t.name NOT IN ({ph})
            GROUP BY t.name ORDER BY n DESC, t.name LIMIT 16""",
        (user["id"], kind, *label_set),
    )
    card_count = db.one(
        """SELECT COUNT(*) AS n FROM cards c JOIN documents d ON d.id = c.document_id
           WHERE d.user_id = ? AND d.kind = ?""",
        (user["id"], kind),
    )["n"]

    # 切换排序 / 翻页 / 点标签时只重渲染列表主体：页头的计数、课程助手卡片都不变，
    # 整页重画会让页面闪一下、滚动位置也丢。
    # 但地址栏里直接敲 ?partial=1 时要给整页 —— 否则用户会看到一个没有样式的裸片段。
    # 地址栏导航的 Sec-Fetch-Dest 是 document，fetch() 发出的是 empty。
    partial = request.query_params.get("partial") == "1"
    if partial and request.headers.get("sec-fetch-dest") == "document":
        partial = False
    return TEMPLATES.TemplateResponse(
        request,
        "_cat_body.html" if partial else "category.html",
        {
            "user": user,
            "kind": kind,
            "kind_label": ingest.KINDS[kind],
            "graph_label": ingest.GRAPH_LABEL.get(kind, "知识图谱"),
            "items": items,
            "page": page,
            "pages": pages,
            "total": total,
            "all_total": all_total,
            "tags": tag_rows,
            "active_tag": tag,
            "sort": sort,
            "q": q,
            "folders": folders,
            "card_count": card_count,
            "kinds": ingest.KINDS,
            "counts": kind_counts(user["id"]),
        },
    )


@app.get("/course/question", response_class=HTMLResponse)
def question_page(request: Request):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return TEMPLATES.TemplateResponse(
        request,
        "study.html",
        {"user": user, "q": "", "result": None, "error": None},
    )


@app.post("/course/question", response_class=HTMLResponse)
def question_analyze(request: Request, q: str = Form("")):
    """课程助手：问题分析。只看课程分类的资料。"""
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    # 字段声明成必填时，空提交会先被判成 422、掉进通用错误页；
    # 改成可选再自己判空，用户能看懂「还没写问题」这句话。
    if not q.strip():
        return TEMPLATES.TemplateResponse(
            request,
            "study.html",
            {"user": user, "q": "", "result": None, "error": "先写一句你卡在哪里"},
        )

    res = study.analyze(user["id"], q)
    return TEMPLATES.TemplateResponse(
        request,
        "study.html",
        {
            "user": user,
            "q": q,
            "result": None if "error" in res else res,
            "error": res.get("error"),
        },
    )


@app.post("/seed")
def load_seed(request: Request):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    seed.load_seed(user["id"])
    return RedirectResponse("/", status_code=303)


# ---------------- 岗位匹配 ----------------
def _jd_samples() -> list[dict]:
    path = config.BASE_DIR / "data" / "jd_samples.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


@app.get("/match", response_class=HTMLResponse)
def match_page(request: Request):
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return TEMPLATES.TemplateResponse(
        request,
        "match.html",
        {"user": user, "jd": "", "result": None, "samples": _jd_samples()},
    )


@app.post("/match", response_class=HTMLResponse)
def match_submit(request: Request, jd: str = Form("")):
    """岗位匹配。

    jd 用默认值而不是 Form(...)：空表单声明为必填时，FastAPI 会回 422 并落到
    通用错误页，用户只看到「请求出错了」，不知道是该填内容。
    留空交给 match.analyze 判「太短」，页面会给出人话。
    """
    user = auth.current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    result = match.analyze(user["id"], jd)
    if result.get("dims"):
        result["ring_svg"] = charts.ring(result["dims"]["total"])
    return TEMPLATES.TemplateResponse(
        request,
        "match.html",
        {"user": user, "jd": jd, "result": result, "samples": _jd_samples()},
    )
