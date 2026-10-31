"""岗位匹配与简历条目生成（F-07，对应设计方案 D-11）。

核心原则：**检索负责找证据，规则负责算分，模型只负责改措辞。**
这样做的好处是每一条结论都能追溯到记录 ID，而且分数可手算复现（AC-07-5）。

任何一步模型不可用，都退回纯规则路径，不阻断出报告。
"""
from __future__ import annotations

import re

from . import db, llm
from .search import keyword_search
from .textproc import keyword_terms

MAX_REQUIREMENTS = 8
MAX_EVIDENCE = 8
MAX_BULLETS = 5

AWARD_WORDS = (
    "特等奖", "一等奖", "二等奖", "三等奖", "金奖", "银奖", "铜奖",
    "冠军", "亚军", "季军", "国家级", "省级", "校级", "院级",
)

REQ_SYS = """你是招聘 JD 解析器。从用户给出的岗位描述中抽取 3-8 条关键要求。
只输出 JSON，格式：
{"requirements":[{"text":"要求（JD 原文短语，去掉序号，不超过20字）","category":"skill|experience|achievement","keywords":["检索用关键词","2到4个"]}]}

rules:
1. 只抽取真正的职责或要求条目。忽略「岗位职责」「任职要求」「工作内容」「职位描述」这类标题行。
2. text 必须去掉开头的编号和标点（如 "1." "①" "-"）。
3. keywords 只放 2-3 个**能被搜索到的具体术语**：技术名、工具名、领域词（如 SQL、Python、数据分析、用户行为）。
   绝对不要放「熟悉」「熟练」「负责」「参与」「一门」「完整」「说明」「类产品」这类动词、量词或无意义残词。
4. category：skill=具体技能工具，experience=相关经历，achievement=成果产出。
不要输出任何解释。"""

BULLET_SYS = """你是简历改写助手。根据给定的一条经历材料，改写成 1-2 句简历条目。
硬性约束：
1. 绝对不能出现材料中没有的数字、奖项名称、时间、单位名。
2. **材料中如果有量化结果（数字、比例、人数、奖项、时间），必须保留，并且沿用材料里的原始写法**（材料写 91% 就写 91%，不要换成「九成」）。
3. 句式是「做了什么 + 结果如何」，不超过 60 字，不用第一人称，不加序号。
4. **如果这条材料与岗位要求明显无关（例如岗位要厨师经验、材料是数据库调优），输出 {"bullet":""}，不要硬凑。**
5. 只输出 JSON：{"bullet":"..."}"""

# 这些词做检索会把大量无关片段召回来，必须挡掉
STOP_KEYWORDS = {
    "熟悉", "熟练", "掌握", "了解", "负责", "参与", "配合", "完成", "能够", "能",
    "使用", "进行", "具备", "具有", "要求", "岗位", "职责", "工作", "内容",
    "优先", "以上", "相关", "良好", "能力", "同学", "独立", "至少",
    "经验", "经历", "基础", "优先者", "者优先", "加分项", "熟练使用",
}

HEADER_LINES = {"岗位职责", "任职要求", "工作内容", "职位描述", "岗位要求", "职责描述"}


def entity_ok(text: str, source: str) -> bool:
    """实体校验：改写后出现的数字与奖项，必须在原文里出现过（AC-07-4）。"""
    for num in re.findall(r"\d+(?:\.\d+)?%?", text):
        if num not in source:
            return False
    for word in AWARD_WORDS:
        if word in text and word not in source:
            return False
    return True


def _fallback_requirements(jd: str) -> list[dict]:
    """模型不可用时的降级：按句读切分，粗分类。"""
    parts = re.split(r"[。\n；;]", jd)
    reqs: list[dict] = []
    for part in parts:
        text = _clean_requirement(part)
        if len(text) < 4 or _is_header(text):
            continue
        if any(k in text for k in ("获奖", "成果", "产出", "成绩")):
            category = "achievement"
        elif any(k in text for k in ("经验", "经历", "项目", "实习")):
            category = "experience"
        else:
            category = "skill"
        words = _clean_keywords(keyword_terms(text)) or keyword_terms(text)[:3]
        reqs.append({"text": text[:40], "category": category, "keywords": words or [text[:8]]})
        if len(reqs) >= MAX_REQUIREMENTS:
            break
    return reqs


def _clean_requirement(text: str) -> str:
    text = re.sub(r"^\s*(?:\d+[.、)）]|[①-⑩]|[-•·*])\s*", "", text)
    return text.strip(" 　:：")


def _clean_keywords(words: list[str]) -> list[str]:
    out: list[str] = []
    for w in words:
        w = str(w).strip()
        if len(w) < 2 or w in STOP_KEYWORDS:
            continue
        if w not in out:
            out.append(w)
    return out[:4]


def _is_header(text: str) -> bool:
    """清洗后仍等于标题词（如「岗位职责」），直接丢弃。"""
    return text in HEADER_LINES


def extract_requirements(jd: str) -> list[dict]:
    if llm.available():
        data = llm.chat_json(
            [
                {"role": "system", "content": REQ_SYS},
                {"role": "user", "content": jd[:3000]},
            ],
            timeout=25,
        )
        raw = (data or {}).get("requirements") if isinstance(data, dict) else None
        if isinstance(raw, list):
            reqs: list[dict] = []
            for item in raw[: MAX_REQUIREMENTS + 4]:
                if not isinstance(item, dict):
                    continue
                text = _clean_requirement(str(item.get("text", "")))
                if len(text) < 4 or _is_header(text):
                    continue
                category = item.get("category")
                if category not in ("skill", "experience", "achievement"):
                    category = "skill"
                keywords = _clean_keywords([str(k) for k in (item.get("keywords") or [])])
                reqs.append(
                    {
                        "text": text[:60],
                        "category": category,
                        "keywords": keywords or keyword_terms(text)[:3] or [text[:8]],
                    }
                )
                if len(reqs) >= MAX_REQUIREMENTS:
                    break
            if reqs:
                return reqs
    return _fallback_requirements(jd)


def _effective_terms(keywords: list[str]) -> list[str]:
    """能用于判断命中的词：长度≥2、非纯数字、不在停用词表。"""
    out: list[str] = []
    for w in keywords:
        w = str(w).strip()
        if len(w) < 2 or w.isdigit() or w in STOP_KEYWORDS:
            continue
        if w not in out:
            out.append(w)
    return out


def collect(user_id: int, requirements: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """逐条要求去档案里找证据。

    命中判定：要求的关键词中，**任意一个有效词**出现在片段里即算命中。
    不再要求「两个词同时命中」—— 实测发现模型给的关键词里常混有噪声词
    （「一门」「说明」「类产品」），双词门槛会造成大面积漏判。
    假命中的问题改由 analyze() 里的全局相关性闸门来解决。
    """
    evidence: list[dict] = []
    gaps: list[dict] = []
    matched_reqs: list[dict] = []
    seen_chunks: set[int] = set()

    for req in requirements:
        query = " ".join(req["keywords"])
        terms = _effective_terms(keyword_terms(query)) or _effective_terms(req["keywords"])
        if not terms:
            gaps.append({"requirement": req["text"], "category": req["category"]})
            continue

        picked = None
        for hit in keyword_search(user_id, query, limit=10):
            text_low = hit["text"].lower()
            if any(t.lower() in text_low for t in terms):
                picked = hit
                break

        if not picked:
            gaps.append({"requirement": req["text"], "category": req["category"]})
            continue

        matched_reqs.append(req)
        if picked["chunk_id"] in seen_chunks:
            continue
        seen_chunks.add(picked["chunk_id"])
        evidence.append(
            {
                "requirement": req["text"],
                "category": req["category"],
                "doc_id": picked["document_id"],
                "chunk_id": picked["chunk_id"],
                "title": picked["title"],
                "text": picked["text"],
            }
        )

    return matched_reqs, evidence[:MAX_EVIDENCE], gaps


def _doc_kinds(doc_ids: set[int]) -> dict[int, str]:
    if not doc_ids:
        return {}
    placeholders = ",".join("?" for _ in doc_ids)
    rows = db.rows(
        f"SELECT id, kind FROM documents WHERE id IN ({placeholders})", tuple(doc_ids)
    )
    return {r["id"]: r["kind"] for r in rows}


def score(requirements: list[dict], matched_reqs: list[dict], evidence: list[dict]) -> dict:
    """规则打分。展示值与总分的算术关系必须能被手算复现（AC-07-5）。"""
    if not requirements or not matched_reqs:
        return {"total": 0, "skill": 0, "experience": 0, "achievement": 0}

    skill_reqs = [r for r in requirements if r["category"] == "skill"] or requirements
    skill_hit = [r for r in matched_reqs if r["category"] == "skill"]
    skill_ratio = len(skill_hit) / max(len(skill_reqs), 1)

    distinct_docs = {e["doc_id"] for e in evidence}
    denom = max(min(len(requirements), 5), 1)
    exp_ratio = min(len(distinct_docs) / denom, 1.0)

    kinds = _doc_kinds(distinct_docs)
    strong = {d for d in distinct_docs if kinds.get(d) in ("award", "project")}
    ach_ratio = (len(strong) / len(distinct_docs)) if distinct_docs else 0.0

    skill = round(skill_ratio * 100)
    experience = round(exp_ratio * 100)
    achievement = round(ach_ratio * 100)
    total = round(0.4 * skill + 0.4 * experience + 0.2 * achievement)

    return {
        "total": total,
        "skill": skill,
        "experience": experience,
        "achievement": achievement,
    }


def _first_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    for sep in ("。", "！", "？"):
        idx = cleaned.find(sep)
        if 0 < idx <= 60:
            return cleaned[: idx + 1]
    return cleaned[:60] + ("…" if len(cleaned) > 60 else "")


def make_bullets(evidence: list[dict]) -> list[dict]:
    bullets: list[dict] = []
    for item in evidence[:MAX_BULLETS]:
        source = item["text"]
        # 实体校验连标题一起比：奖项常写在标题里，正文只写「拿了省二」
        source_for_check = f"{item['title']}\n{source}"
        if llm.available():
            data = llm.chat_json(
                [
                    {"role": "system", "content": BULLET_SYS},
                    {
                        "role": "user",
                        "content": f"材料：{source}\n\n岗位要求：{item['requirement']}",
                    },
                ],
                timeout=25,
            )
            text = str((data or {}).get("bullet", "")).strip() if isinstance(data, dict) else ""
            # 模型判定不相关，或改写出原文没有的实体 → 丢弃该条，绝不硬凑
            if not text or not entity_ok(text, source_for_check):
                continue
        else:
            text = _first_sentence(source)

        bullets.append(
            {
                "text": text[:120],
                "doc_id": item["doc_id"],
                "chunk_id": item["chunk_id"],
                "title": item["title"],
            }
        )
    return bullets


def analyze(user_id: int, jd: str) -> dict:
    jd = (jd or "").strip()
    if len(jd) < 10:
        return {"error": "岗位描述太短，至少 10 个字。"}

    requirements = extract_requirements(jd)
    if not requirements:
        return {"error": "没能从这段描述里识别出要求，试试粘贴岗位职责或任职要求部分。"}

    matched, evidence, gaps = collect(user_id, requirements)

    # 全局相关性闸门：整段 JD 只有零星命中时，说明档案与这个岗位基本无关。
    # 逐条卡门槛会被关键词噪声击穿，看整体命中率更稳定。
    # 双条件是为了避免"33% 恰好卡在阈值边缘"这种脆弱判定：
    # 覆盖率过低，或者只有零星命中且证据全来自同一份记录，都判为不相关。
    coverage = len(matched) / max(len(requirements), 1)
    distinct_docs = len({e["doc_id"] for e in evidence})
    unrelated = coverage < 0.34 or (distinct_docs <= 1 and coverage < 0.6)

    if unrelated:
        dims = {"total": 0, "skill": 0, "experience": 0, "achievement": 0}
        bullets: list[dict] = []
    else:
        dims = score(requirements, matched, evidence)
        bullets = make_bullets(evidence)

    return {
        "requirements": requirements,
        "evidence": evidence,
        "gaps": gaps,
        "dims": dims,
        "bullets": bullets,
        "unrelated": unrelated,
        "coverage": round(coverage * 100),
        "degraded": not llm.available(),
    }
