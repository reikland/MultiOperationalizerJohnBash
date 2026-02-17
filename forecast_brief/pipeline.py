from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from forecast_brief.prompts_core import SYSTEM_T2, USER_T2
from forecast_brief.prompts_questions import SYSTEM_Q, USER_Q
from forecast_brief.utils import (
    all_blocks_have_required_fields,
    clean,
    count_question_blocks,
    extract_between,
    looks_internal_metric_question,
    split_question_blocks,
    ensure_online_model,
)

LLMCallable = Callable[..., str]

def run_tier2_three_calls(
    llm: LLMCallable,
    api_key: str,
    sector_model_raw: str,
    company: str,
    tier1_text: str,
    geo: str,
    horizon: int,
    temperature: float,
    max_tokens: int,
    stream_to: Optional[object] = None,
) -> Tuple[str, str]:
    sector_model = ensure_online_model(sector_model_raw)
    prior_snapshot = "—"
    call_texts: List[str] = []

    for call_idx in (1, 2, 3):
        text = clean(
            llm(
                api_key=api_key,
                model=sector_model,
                system=SYSTEM_T2.format(call_idx=call_idx, call_total=3),
                user=USER_T2.format(
                    call_idx=call_idx,
                    call_total=3,
                    company=company,
                    tier1_text=tier1_text,
                    geo=geo,
                    horizon=horizon,
                    prior_snapshot=prior_snapshot,
                ),
                temperature=max(0.15, temperature),
                max_tokens=max_tokens,
                stream_to=stream_to,
            )
        )
        snapshot = extract_between(
            text,
            "UPDATED SNAPSHOT (use this whole block for downstream tiers):",
            f"END TIER 2 CALL {call_idx}",
        )
        prior_snapshot = snapshot.strip() if snapshot else prior_snapshot
        call_texts.append(text)

    return "\n\n".join(call_texts).strip(), prior_snapshot.strip() if prior_snapshot else "—"

def run_questions_per_cluster(
    llm: LLMCallable,
    api_key: str,
    question_model: str,
    company: str,
    tier1_text: str,
    tier2_snapshot: str,
    clusters: List[Tuple[str, str]],
    geo: str,
    horizon: int,
    risk: str,
    deadline_date: str,
    temperature: float,
    max_tokens: int,
    stream_to: Optional[object] = None,
) -> str:
    outputs: List[str] = []
    for cluster_name, cluster_block in clusters:
        raw = clean(
            llm(
                api_key=api_key,
                model=question_model,
                system=SYSTEM_Q,
                user=USER_Q.format(
                    deadline_date=deadline_date,
                    company=company,
                    tier1_text=tier1_text,
                    tier2_snapshot=tier2_snapshot,
                    cluster_name=cluster_name,
                    cluster_block=cluster_block,
                    geo=geo,
                    horizon=horizon,
                    risk=risk,
                ),
                temperature=temperature,
                max_tokens=max_tokens,
                stream_to=stream_to,
            )
        )
        if _needs_repair(raw):
            raw = clean(
                llm(
                    api_key=api_key,
                    model=question_model,
                    system="You are a strict forecasting editor and format enforcer. Output RAW TEXT only.",
                    user=_repair_prompt(deadline_date, cluster_name, company, tier1_text, tier2_snapshot, cluster_block, raw),
                    temperature=0.2,
                    max_tokens=min(2000, max_tokens),
                    stream_to=None,
                )
            )
        outputs.append(raw)
    return "\n\n".join(outputs).strip()

def _needs_repair(raw: str) -> bool:
    n_questions = count_question_blocks(raw)
    return (
        n_questions < 2
        or n_questions > 3
        or not split_question_blocks(raw)
        or not all_blocks_have_required_fields(raw)
        or looks_internal_metric_question(raw)
    )

def _repair_prompt(
    deadline_date: str,
    cluster_name: str,
    company: str,
    tier1_text: str,
    tier2_snapshot: str,
    cluster_block: str,
    raw: str,
) -> str:
    return f"""Rewrite the output so that:
1) It contains EXACTLY 2 OR 3 DISTINCT question blocks (not duplicates).
2) EACH block matches the EXACT required structure and includes all required fields.
3) ALL questions are PUBLICLY RESOLVABLE (no internal company metrics, no ARR/churn/customer-base measures).
4) EACH question remains decision-useful for Strategy A (compliance/PDP) vs Strategy B (payments/finance).
5) EACH question includes a PUBLIC resolution hook in Answer Guidance (dataset/page/type; no URL needed).
Use DEADLINE_DATE {deadline_date} and do not exceed it.
Cluster name: {cluster_name}

IMPORTANT CONTEXT (must be used, do not ignore):
Company description:
{company}

Tier 1 output:
{tier1_text}

Tier 2 UPDATED SNAPSHOT:
{tier2_snapshot}

Cluster block:
{cluster_block}

Faulty output:
{raw}
"""
