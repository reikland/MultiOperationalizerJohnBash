from __future__ import annotations

import re
import textwrap
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple


def today_utc_date() -> date:
    return datetime.utcnow().date()


def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def last_day_of_month(year: int, month: int) -> int:
    first = date(year, month, 1)
    next_first = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return (next_first - timedelta(days=1)).day


def add_months(current: date, months: int) -> date:
    year = current.year + (current.month - 1 + months) // 12
    month = (current.month - 1 + months) % 12 + 1
    return date(year, month, min(current.day, last_day_of_month(year, month)))


def clean(text: str) -> str:
    return (text or "").strip()


def wrap(text: str, width: int = 96) -> str:
    lines: List[str] = []
    for paragraph in (text or "").splitlines():
        lines.extend([""] if not paragraph.strip() else textwrap.fill(paragraph, width=width).splitlines())
    return "\n".join(lines).strip()


def ensure_nonempty_geo(geo: str) -> str:
    return clean(geo) or "Global"


def ensure_online_model(model: str) -> str:
    clean_model = clean(model)
    if not clean_model:
        return "openai/gpt-5.2-chat:online"
    return clean_model if clean_model.endswith(":online") else f"{clean_model}:online"


def extract_between(text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
    if not text:
        return ""
    start_idx = text.find(start_marker)
    if start_idx < 0:
        return ""
    tail = text[start_idx + len(start_marker) :]
    if end_marker:
        end_idx = tail.find(end_marker)
        if end_idx >= 0:
            tail = tail[:end_idx]
    return tail.strip()


def split_clusters(tier3_text: str) -> List[Tuple[str, str]]:
    starts = [
        (match.start(), match.group(1).strip())
        for match in re.finditer(r"^CLUSTER:\s*(.+?)\s*$", tier3_text or "", flags=re.MULTILINE)
    ]
    blocks: List[Tuple[str, str]] = []
    for idx, (pos, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(tier3_text)
        block = (tier3_text or "")[pos:end].strip()
        if name and block:
            blocks.append((name, block))
    return blocks


def split_question_blocks(qtext: str) -> List[str]:
    starts = [
        match.start()
        for match in re.finditer(r"^QUESTION FOR CLUSTER:\s*.+\s*$", qtext or "", flags=re.MULTILINE)
    ]
    return [
        (qtext or "")[start : (starts[idx + 1] if idx + 1 < len(starts) else len(qtext or ""))].strip()
        for idx, start in enumerate(starts)
        if (qtext or "")[start : (starts[idx + 1] if idx + 1 < len(starts) else len(qtext or ""))].strip()
    ]


def count_question_blocks(qtext: str) -> int:
    return len(re.findall(r"^QUESTION FOR CLUSTER:\s*.+$", qtext or "", flags=re.MULTILINE))


def has_required_question_fields(qtext: str) -> bool:
    required = [
        "QUESTION FOR CLUSTER:",
        "TITLE:",
        "TYPE:",
        "QUESTION (one sentence):",
        "BACKGROUND",
        "CLUSTER CONTEXT",
        "FERMI STARTING POINT",
        "WHAT TO CONSIDER",
        "ANSWER GUIDANCE",
        "AMBIGUITY CHECKS",
    ]
    return all(field in (qtext or "") for field in required)


def all_blocks_have_required_fields(qtext: str) -> bool:
    blocks = split_question_blocks(qtext)
    return bool(blocks) and all(has_required_question_fields(block) for block in blocks)


def looks_internal_metric_question(qtext: str) -> bool:
    bad_phrases = [
        "company’s", "company's", "our", "customer base", "by arr", "arr", "nrr", "churn",
        "gross margin", "cac", "sales cycle", "internal", "audited", "go-live across customers",
        "net interest margin on embedded invoice financing", "new arr", "contribution margin attributable",
        "percentage of invoices processed in the prior 90 days across the company’s active customer base",
        "documented evidence", "at least 30% of the company",
    ]
    lower_text = (qtext or "").lower()
    return any(phrase.lower() in lower_text for phrase in bad_phrases)
